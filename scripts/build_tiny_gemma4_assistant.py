#!/usr/bin/env python3
"""Build the tiny Gemma4 ``*-assistant`` shared-KV speculative fixture.

Run from the onnx-genai repository root::

    python3 scripts/build_tiny_gemma4_assistant.py

Unlike the other fixtures this one is built with the raw ONNX IR API
(:mod:`onnx_ir`) rather than Mobius, so it needs no external exporter. It emits
two graphs under ``tests/fixtures/tiny-gemma4-assistant``:

* ``model.onnx.textproto`` -- a tiny *target* decoder. Its logits and hidden state are a
  deterministic function of the current token only (a Gather into random but
  fixed tables), so plain greedy decoding is fully reproducible. It carries a
  two-layer paged KV cache (layer 0 stands in for Gemma4's sliding-attention
  layers, layer 1 for the full-attention layers) whose *values* are irrelevant
  to the logits -- only the growing shapes matter for the runtime plumbing.
  It also exposes ``hidden_states.0`` (Float32 ``[B, S, H]``) for seeding the
  assistant.
* ``assistant/model.onnx.textproto`` -- a tiny Gemma4 assistant matching the shared-KV
  contract: ``inputs_embeds`` ``[B, q, 2*H]``, ``position_ids``,
  ``attention_mask`` and ``shared_kv.{sliding,full}_attention.{key,value}``
  inputs; ``logits`` ``[B, q, V]`` and ``projected_state`` ``[B, q, H]``
  outputs. It genuinely reads the shared-KV tensors (they feed a
  multiplied-by-zero term) so the binding path is exercised, while remaining a
  deterministic function of ``inputs_embeds``.

The models are intentionally tiny and only validate graph/runtime contracts;
they are not meaningful language models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir

SEED = 20260713
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures" / "tiny-gemma4-assistant"

VOCAB = 32
HIDDEN = 16
KV_HEADS = 2
HEAD_DIM = 8
NUM_LAYERS = 2

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(SEED + sum(ord(ch) for ch in tag))


def _initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.tensor(array.astype(np.float32), name=name))


def _int_initializer(name: str, array: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.tensor(array.astype(np.int64), name=name))


def _node(op_type, inputs, out_specs, attrs=None, name=None):
    """Create a node and return its named/typed output Values."""
    outputs = [ir.val(out_name, dtype, shape) for (out_name, dtype, shape) in out_specs]
    node = ir.node(op_type, inputs=inputs, attributes=attrs or {}, outputs=outputs, name=name)
    return node, outputs


def build_target() -> ir.Model:
    input_ids = ir.val("input_ids", INT64, ["batch", "sequence_len"])
    attention_mask = ir.val("attention_mask", INT64, ["batch", "total_seq_len"])
    position_ids = ir.val("position_ids", INT64, ["batch", "sequence_len"])

    past_inputs = []
    for layer in range(NUM_LAYERS):
        past_inputs.append(
            ir.val(f"past_key_values.{layer}.key", FLOAT, ["batch", KV_HEADS, "past_seq", HEAD_DIM])
        )
        past_inputs.append(
            ir.val(f"past_key_values.{layer}.value", FLOAT, ["batch", KV_HEADS, "past_seq", HEAD_DIM])
        )

    lm_table = _initializer("lm_table", _rng("lm").standard_normal((VOCAB, VOCAB)))
    hidden_table = _initializer(
        "hidden_table", _rng("hidden").standard_normal((VOCAB, HIDDEN)) * 0.02
    )
    two_const = _int_initializer("kv_heads_const", np.array([KV_HEADS]))
    head_dim_const = _int_initializer("head_dim_const", np.array([HEAD_DIM]))
    idx0 = _int_initializer("idx0", np.array([0]))
    idx1 = _int_initializer("idx1", np.array([1]))

    nodes = []

    _, (logits,) = _node(
        "Gather",
        [lm_table, input_ids],
        [("logits", FLOAT, ["batch", "sequence_len", VOCAB])],
        {"axis": 0},
        name="target_logits",
    )
    nodes.append(logits.producer())

    _, (hidden,) = _node(
        "Gather",
        [hidden_table, input_ids],
        [("hidden_states.0", FLOAT, ["batch", "sequence_len", HIDDEN])],
        {"axis": 0},
        name="target_hidden",
    )
    nodes.append(hidden.producer())

    # Build a zero-valued [B, KV_HEADS, S, HEAD_DIM] tensor for the new KV rows.
    _, (shape_ids,) = _node(
        "Shape", [input_ids], [("ids_shape", INT64, [2])], name="ids_shape"
    )
    nodes.append(shape_ids.producer())
    _, (batch_dim,) = _node(
        "Gather", [shape_ids, idx0], [("batch_dim", INT64, [1])], {"axis": 0}, name="batch_dim"
    )
    nodes.append(batch_dim.producer())
    _, (seq_dim,) = _node(
        "Gather", [shape_ids, idx1], [("seq_dim", INT64, [1])], {"axis": 0}, name="seq_dim"
    )
    nodes.append(seq_dim.producer())
    _, (new_shape,) = _node(
        "Concat",
        [batch_dim, two_const, seq_dim, head_dim_const],
        [("new_kv_shape", INT64, [4])],
        {"axis": 0},
        name="new_kv_shape",
    )
    nodes.append(new_shape.producer())
    _, (new_kv,) = _node(
        "ConstantOfShape",
        [new_shape],
        [("new_kv_zeros", FLOAT, ["batch", KV_HEADS, "sequence_len", HEAD_DIM])],
        {"value": ir.tensor(np.zeros((1,), dtype=np.float32), name="new_kv_fill")},
        name="new_kv_zeros",
    )
    nodes.append(new_kv.producer())

    outputs = [logits]
    for layer in range(NUM_LAYERS):
        past_key = past_inputs[2 * layer]
        past_value = past_inputs[2 * layer + 1]
        _, (present_key,) = _node(
            "Concat",
            [past_key, new_kv],
            [(f"present.{layer}.key", FLOAT, ["batch", KV_HEADS, "total_seq_len", HEAD_DIM])],
            {"axis": 2},
            name=f"present_{layer}_key",
        )
        nodes.append(present_key.producer())
        _, (present_value,) = _node(
            "Concat",
            [past_value, new_kv],
            [(f"present.{layer}.value", FLOAT, ["batch", KV_HEADS, "total_seq_len", HEAD_DIM])],
            {"axis": 2},
            name=f"present_{layer}_value",
        )
        nodes.append(present_value.producer())
        outputs.append(present_key)
        outputs.append(present_value)
    outputs.append(hidden)

    graph = ir.Graph(
        inputs=[input_ids, attention_mask, position_ids, *past_inputs],
        outputs=outputs,
        nodes=nodes,
        initializers=[
            lm_table,
            hidden_table,
            two_const,
            head_dim_const,
            idx0,
            idx1,
        ],
        opset_imports={"": 24},
        name="tiny_gemma4_target",
    )
    return ir.Model(graph, ir_version=11, producer_name="onnx-genai-tests")


def build_assistant() -> ir.Model:
    inputs_embeds = ir.val("inputs_embeds", FLOAT, ["batch", "q", 2 * HIDDEN])
    position_ids = ir.val("position_ids", INT64, ["batch", "q"])
    attention_mask = ir.val("attention_mask", INT64, ["batch", "kv_len"])

    shared_inputs = []
    for group in ("sliding_attention", "full_attention"):
        shared_inputs.append(
            ir.val(f"shared_kv.{group}.key", FLOAT, ["batch", KV_HEADS, "kv_len", HEAD_DIM])
        )
        shared_inputs.append(
            ir.val(f"shared_kv.{group}.value", FLOAT, ["batch", KV_HEADS, "kv_len", HEAD_DIM])
        )

    w_logits = _initializer("assistant_w", _rng("assist").standard_normal((HIDDEN, VOCAB)) * 0.1)
    slice_starts = _int_initializer("slice_starts", np.array([HIDDEN]))
    slice_ends = _int_initializer("slice_ends", np.array([2 * HIDDEN]))
    slice_axes = _int_initializer("slice_axes", np.array([2]))
    zero_scalar = _initializer("assist_zero", np.array(0.0))

    nodes = []

    # cur = inputs_embeds[..., HIDDEN:2*HIDDEN]  (the current projected state)
    _, (cur,) = _node(
        "Slice",
        [inputs_embeds, slice_starts, slice_ends, slice_axes],
        [("cur_state", FLOAT, ["batch", "q", HIDDEN])],
        name="slice_cur",
    )
    nodes.append(cur.producer())

    # projected_state threads the current state forward unchanged.
    _, (projected,) = _node(
        "Identity",
        [cur],
        [("projected_state", FLOAT, ["batch", "q", HIDDEN])],
        name="projected_state",
    )
    nodes.append(projected.producer())

    _, (base_logits,) = _node(
        "MatMul",
        [cur, w_logits],
        [("base_logits", FLOAT, ["batch", "q", VOCAB])],
        name="assistant_matmul",
    )
    nodes.append(base_logits.producer())

    # Reference every shared_kv input so the binding is genuinely consumed, then
    # multiply by zero so it does not perturb the deterministic logits.
    shared_sum = None
    for idx, shared in enumerate(shared_inputs):
        _, (reduced,) = _node(
            "ReduceSum",
            [shared],
            [(f"shared_reduce_{idx}", FLOAT, [])],
            {"keepdims": 0},
            name=f"shared_reduce_{idx}",
        )
        nodes.append(reduced.producer())
        if shared_sum is None:
            shared_sum = reduced
        else:
            _, (acc,) = _node(
                "Add",
                [shared_sum, reduced],
                [(f"shared_acc_{idx}", FLOAT, [])],
                name=f"shared_acc_{idx}",
            )
            nodes.append(acc.producer())
            shared_sum = acc

    _, (zero_term,) = _node(
        "Mul",
        [shared_sum, zero_scalar],
        [("shared_zero_term", FLOAT, [])],
        name="shared_zero_term",
    )
    nodes.append(zero_term.producer())

    _, (logits,) = _node(
        "Add",
        [base_logits, zero_term],
        [("logits", FLOAT, ["batch", "q", VOCAB])],
        name="assistant_logits",
    )
    nodes.append(logits.producer())

    graph = ir.Graph(
        inputs=[inputs_embeds, position_ids, attention_mask, *shared_inputs],
        outputs=[logits, projected],
        nodes=nodes,
        initializers=[w_logits, slice_starts, slice_ends, slice_axes, zero_scalar],
        opset_imports={"": 24},
        name="tiny_gemma4_assistant",
    )
    return ir.Model(graph, ir_version=11, producer_name="onnx-genai-tests")


def _copy_tokenizer(dest: Path) -> None:
    source = ROOT / "tests" / "fixtures" / "tiny-llm" / "tokenizer.json"
    dest.write_text(source.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assistant_dir = OUT_DIR / "assistant"
    assistant_dir.mkdir(parents=True, exist_ok=True)

    target = build_target()
    assistant = build_assistant()

    ir.save(target, str(OUT_DIR / "model.onnx.textproto"), format="textproto")
    ir.save(assistant, str(assistant_dir / "model.onnx.textproto"), format="textproto")
    _copy_tokenizer(OUT_DIR / "tokenizer.json")

    # The shared-KV proposer builds inputs_embeds from the *target* input-token
    # embedding of the last token (concat(embed(last_token), hidden)). This tiny
    # target uses `hidden_table` as its embedding, so export it as the raw
    # little-endian f32 [VOCAB, HIDDEN] table the runtime looks tokens up in.
    input_embedding = (_rng("hidden").standard_normal((VOCAB, HIDDEN)) * 0.02).astype(
        "<f4"
    )
    (OUT_DIR / "input_embedding.f32").write_bytes(input_embedding.tobytes())

    manifest = {
        "generator": "scripts/build_tiny_gemma4_assistant.py",
        "seed": SEED,
        "vocab_size": VOCAB,
        "backbone_hidden_size": HIDDEN,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_target_layers": NUM_LAYERS,
        "shared_kv": [
            {"name": "sliding_attention", "target_layers": [0]},
            {"name": "full_attention", "target_layers": [1]},
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
