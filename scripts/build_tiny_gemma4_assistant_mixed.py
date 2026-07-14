#!/usr/bin/env python3
"""Build the tiny Gemma4 ``*-assistant`` shared-KV fixture with MIXED per-layer head_dim.

Run from the onnx-genai repository root::

    python3 scripts/build_tiny_gemma4_assistant_mixed.py

This is the companion to ``build_tiny_gemma4_assistant.py`` (the uniform-head_dim
fixture). Unlike the uniform fixture (``HEAD_DIM=8`` for every layer), this one
mirrors Gemma-4 E2B's actual geometry at tiny scale:

* ``sliding_attention`` layer (layer 0): ``head_dim=8``  (mirrors E2B's 256)
* ``full_attention``    layer (layer 1): ``head_dim=16`` (mirrors E2B's 512)

The mixed-head_dim fixture is the coverage vehicle for Sapper's blocker #4 (per-layer
paged-cache geometry). The existing uniform fixture in ``tiny-gemma4-assistant`` is
**not changed**, so all existing tests continue to pass.

The fixture is written to ``tests/fixtures/tiny-gemma4-assistant-mixed``. The
associated W5b Rust token-identity test is guarded with
``#[ignore = "enable after W3 per-layer paged cache lands"]`` and lives in
``crates/onnx-genai-engine/tests/gemma4_assistant_full.rs``.

Layer → head_dim map (used by W5b assertions)
---------------------------------------------
Layer 0 (sliding_attention, target_layers=[0]):
    past_key_values.0.key/value: [batch, 2, past_seq, 8]
    present.0.key/value:         [batch, 2, total_seq_len, 8]

Layer 1 (full_attention, target_layers=[1]):
    past_key_values.1.key/value: [batch, 2, past_seq, 16]
    present.1.key/value:         [batch, 2, total_seq_len, 16]

Assistant shared-KV inputs:
    shared_kv.sliding_attention.key/value: [batch, 2, kv_len, 8]
    shared_kv.full_attention.key/value:    [batch, 2, kv_len, 16]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir

SEED = 20260713
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures" / "tiny-gemma4-assistant-mixed"

VOCAB = 32
HIDDEN = 16
KV_HEADS = 2
# Per-layer head dims — mirroring E2B's 256(sliding) / 512(full) at tiny scale
SLIDING_HEAD_DIM = 8
FULL_HEAD_DIM = 16
NUM_LAYERS = 2  # layer 0 = sliding, layer 1 = full

FLOAT = ir.DataType.FLOAT
INT64 = ir.DataType.INT64

# Map layer index → (group name, head_dim)
LAYER_CONFIG = [
    ("sliding_attention", SLIDING_HEAD_DIM),
    ("full_attention", FULL_HEAD_DIM),
]


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
    """Build the tiny target decoder with heterogeneous per-layer KV head_dim.

    Layer 0 (sliding_attention): KV head_dim = SLIDING_HEAD_DIM (8)
    Layer 1 (full_attention):    KV head_dim = FULL_HEAD_DIM (16)

    KV output shapes:
        present.0.key/value: [batch, KV_HEADS, total_seq_len, 8]
        present.1.key/value: [batch, KV_HEADS, total_seq_len, 16]
    """
    input_ids = ir.val("input_ids", INT64, ["batch", "sequence_len"])
    attention_mask = ir.val("attention_mask", INT64, ["batch", "total_seq_len"])
    position_ids = ir.val("position_ids", INT64, ["batch", "sequence_len"])

    past_inputs = []
    for layer, (_, head_dim) in enumerate(LAYER_CONFIG):
        past_inputs.append(
            ir.val(
                f"past_key_values.{layer}.key",
                FLOAT,
                ["batch", KV_HEADS, "past_seq", head_dim],
            )
        )
        past_inputs.append(
            ir.val(
                f"past_key_values.{layer}.value",
                FLOAT,
                ["batch", KV_HEADS, "past_seq", head_dim],
            )
        )

    lm_table = _initializer("lm_table", _rng("lm").standard_normal((VOCAB, VOCAB)))
    hidden_table = _initializer(
        "hidden_table", _rng("hidden").standard_normal((VOCAB, HIDDEN)) * 0.02
    )
    kv_heads_const = _int_initializer("kv_heads_const", np.array([KV_HEADS]))
    idx0 = _int_initializer("idx0", np.array([0]))
    idx1 = _int_initializer("idx1", np.array([1]))
    # Per-layer head_dim constants (one for each distinct dim)
    sliding_hd_const = _int_initializer(
        "sliding_head_dim_const", np.array([SLIDING_HEAD_DIM])
    )
    full_hd_const = _int_initializer("full_head_dim_const", np.array([FULL_HEAD_DIM]))

    layer_hd_consts = [sliding_hd_const, full_hd_const]

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

    # Extract batch and sequence dims from input_ids shape
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

    outputs = [logits]
    for layer, (group_name, head_dim) in enumerate(LAYER_CONFIG):
        hd_const = layer_hd_consts[layer]

        # Build zero-valued [B, KV_HEADS, S, head_dim] tensor for new KV rows
        _, (new_shape,) = _node(
            "Concat",
            [batch_dim, kv_heads_const, seq_dim, hd_const],
            [(f"new_kv_shape_{layer}", INT64, [4])],
            {"axis": 0},
            name=f"new_kv_shape_{layer}",
        )
        nodes.append(new_shape.producer())
        _, (new_kv,) = _node(
            "ConstantOfShape",
            [new_shape],
            [(f"new_kv_zeros_{layer}", FLOAT, ["batch", KV_HEADS, "sequence_len", head_dim])],
            {"value": ir.tensor(np.zeros((1,), dtype=np.float32), name=f"new_kv_fill_{layer}")},
            name=f"new_kv_zeros_{layer}",
        )
        nodes.append(new_kv.producer())

        past_key = past_inputs[2 * layer]
        past_value = past_inputs[2 * layer + 1]
        _, (present_key,) = _node(
            "Concat",
            [past_key, new_kv],
            [(f"present.{layer}.key", FLOAT, ["batch", KV_HEADS, "total_seq_len", head_dim])],
            {"axis": 2},
            name=f"present_{layer}_key",
        )
        nodes.append(present_key.producer())
        _, (present_value,) = _node(
            "Concat",
            [past_value, new_kv],
            [(f"present.{layer}.value", FLOAT, ["batch", KV_HEADS, "total_seq_len", head_dim])],
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
            kv_heads_const,
            idx0,
            idx1,
            sliding_hd_const,
            full_hd_const,
        ],
        opset_imports={"": 24},
        name="tiny_gemma4_target_mixed",
    )
    return ir.Model(graph, ir_version=11, producer_name="onnx-genai-tests")


def build_assistant() -> ir.Model:
    """Build the tiny assistant with heterogeneous per-group shared-KV shapes.

    Shared-KV inputs:
        shared_kv.sliding_attention.key/value: [batch, KV_HEADS, kv_len, SLIDING_HEAD_DIM=8]
        shared_kv.full_attention.key/value:    [batch, KV_HEADS, kv_len, FULL_HEAD_DIM=16]
    """
    inputs_embeds = ir.val("inputs_embeds", FLOAT, ["batch", "q", 2 * HIDDEN])
    position_ids = ir.val("position_ids", INT64, ["batch", "q"])
    attention_mask = ir.val("attention_mask", INT64, ["batch", "kv_len"])

    # Heterogeneous shared-KV inputs — each group uses its own head_dim
    shared_inputs = []
    for group_name, head_dim in LAYER_CONFIG:
        shared_inputs.append(
            ir.val(
                f"shared_kv.{group_name}.key",
                FLOAT,
                ["batch", KV_HEADS, "kv_len", head_dim],
            )
        )
        shared_inputs.append(
            ir.val(
                f"shared_kv.{group_name}.value",
                FLOAT,
                ["batch", KV_HEADS, "kv_len", head_dim],
            )
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

    # projected_state threads the current state forward unchanged
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

    # Reference every shared_kv input (multiply by zero so logits stay deterministic)
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
        name="tiny_gemma4_assistant_mixed",
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

    ir.save(target, str(OUT_DIR / "model.onnx"))
    ir.save(assistant, str(assistant_dir / "model.onnx"))
    _copy_tokenizer(OUT_DIR / "tokenizer.json")

    # The shared-KV proposer builds inputs_embeds from the target input-token
    # embedding of the last token (concat(embed(last_token), hidden)). The tiny
    # target uses `hidden_table` as its embedding — export it as raw little-endian
    # f32 [VOCAB, HIDDEN] so the engine's LinearEmbedder / read_f32_weights loads it.
    input_embedding = (_rng("hidden").standard_normal((VOCAB, HIDDEN)) * 0.02).astype("<f4")
    (OUT_DIR / "input_embedding.f32").write_bytes(input_embedding.tobytes())

    manifest = {
        "generator": "scripts/build_tiny_gemma4_assistant_mixed.py",
        "seed": SEED,
        "vocab_size": VOCAB,
        "backbone_hidden_size": HIDDEN,
        "kv_heads": KV_HEADS,
        "head_dim_per_layer": {
            "0": {"group": "sliding_attention", "head_dim": SLIDING_HEAD_DIM},
            "1": {"group": "full_attention", "head_dim": FULL_HEAD_DIM},
        },
        "num_target_layers": NUM_LAYERS,
        "shared_kv": [
            {"name": "sliding_attention", "target_layers": [0]},
            {"name": "full_attention", "target_layers": [1]},
        ],
        "kv_output_shapes": {
            "present.0.key": ["batch", KV_HEADS, "total_seq_len", SLIDING_HEAD_DIM],
            "present.0.value": ["batch", KV_HEADS, "total_seq_len", SLIDING_HEAD_DIM],
            "present.1.key": ["batch", KV_HEADS, "total_seq_len", FULL_HEAD_DIM],
            "present.1.value": ["batch", KV_HEADS, "total_seq_len", FULL_HEAD_DIM],
        },
        "assistant_shared_kv_shapes": {
            "shared_kv.sliding_attention.key": ["batch", KV_HEADS, "kv_len", SLIDING_HEAD_DIM],
            "shared_kv.sliding_attention.value": ["batch", KV_HEADS, "kv_len", SLIDING_HEAD_DIM],
            "shared_kv.full_attention.key": ["batch", KV_HEADS, "kv_len", FULL_HEAD_DIM],
            "shared_kv.full_attention.value": ["batch", KV_HEADS, "kv_len", FULL_HEAD_DIM],
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote mixed-head_dim fixture to {OUT_DIR}")
    print()
    print("Layer → head_dim map (for W5b assertions):")
    for layer, (group, hd) in enumerate(LAYER_CONFIG):
        print(f"  Layer {layer} ({group}): head_dim={hd}")
    print()
    print("KV output shapes:")
    for name, shape in manifest["kv_output_shapes"].items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
