#!/usr/bin/env python3
"""Generate the attention / MoE / KV-cache assignment fixtures.

    python3 tests/fixtures/generate_attention_fixtures.py

These exist for one assertion: when this EP is loaded it must take every node
it supports, and ORT's CPU EP must be left with nothing. `plugin_ort_e2e.rs`
runs each fixture twice -- once reading ORT's node-to-EP assignment back, and
once with `session.disable_cpu_ep_fallback=1` so an unclaimed node fails
session creation outright.

The suite that already existed covered activations and normalisation only, so
the guarantee was unproven for exactly the operators this EP is built for.
Every graph here is single-node where the operator allows it, because a
multi-node graph cannot distinguish "the node was claimed" from "the node was
swallowed by a partition around it".

Shapes are decode-sized on purpose: that is the range the assignment argument
used to be made about, so it is the range the fixtures have to cover.
"""
import os

import onnx
from onnx import TensorProto, helper
from google.protobuf import text_format

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

MS = "com.microsoft"


def save(model, name, check=True):
    path = os.path.join(FIXTURES_DIR, name, "model.onnx.textproto")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if check:
        onnx.checker.check_model(model)
    with open(path, "w") as f:
        f.write(text_format.MessageToString(model))
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")


def vi(name, dims, elem=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, elem, dims)


def finish(nodes, name, inputs, outputs, opset=17, ms=False, inits=None):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits or [])
    imports = [helper.make_opsetid("", opset)]
    if ms:
        imports.append(helper.make_opsetid(MS, 1))
    model = helper.make_model(graph, opset_imports=imports)
    model.ir_version = 10
    return model


# ── Softmax ──────────────────────────────────────────────────────────────────
# The attention softmax shape: [batch=1, heads=32, q=1, kv=1024], reduced over
# the last axis. Kept single-node so the claim cannot be inherited from a
# neighbour.
def gen_softmax_assignment_f32():
    X = vi("X", [1, 32, 1, 1024])
    Z = vi("Z", [1, 32, 1, 1024])
    node = helper.make_node("Softmax", ["X"], ["Z"], axis=-1)
    save(finish([node], "softmax_assignment_f32", [X], [Z], opset=13),
         "softmax_assignment_f32")


# ── Transpose ────────────────────────────────────────────────────────────────
# The QK layout permutation [B,S,H,D] -> [B,H,S,D], which is the one that
# actually appears between a projection and a QK GEMM.
def gen_transpose_assignment_f32():
    X = vi("X", [1, 128, 32, 128])
    Z = vi("Z", [1, 32, 128, 128])
    node = helper.make_node("Transpose", ["X"], ["Z"], perm=[0, 2, 1, 3])
    save(finish([node], "transpose_assignment_f32", [X], [Z], opset=13),
         "transpose_assignment_f32")


# ── Concat (KV cache append) ─────────────────────────────────────────────────
# past [B, H, P, D] ++ new [B, H, 1, D] along the sequence axis: the decode-step
# KV cache growth, which is where the Concat cost in this EP actually lives.
def gen_kv_concat_assignment_f32():
    past = vi("past", [1, 8, 1023, 128])
    cur = vi("cur", [1, 8, 1, 128])
    out = vi("present", [1, 8, 1024, 128])
    node = helper.make_node("Concat", ["past", "cur"], ["present"], axis=2)
    save(finish([node], "kv_concat_assignment_f32", [past, cur], [out], opset=13),
         "kv_concat_assignment_f32")


# ── ScatterND (in-place KV cache update) ─────────────────────────────────────
# The other KV-cache idiom: write the new token into a preallocated cache
# instead of growing it. Output shape == data shape, which is the rule that was
# missing from the plugin's shape table.
def gen_kv_scatternd_assignment_f32():
    data = vi("cache", [1, 8, 1024, 128])
    indices = vi("indices", [1, 3], TensorProto.INT64)
    updates = vi("updates", [1, 128])
    out = vi("updated", [1, 8, 1024, 128])
    node = helper.make_node("ScatterND", ["cache", "indices", "updates"], ["updated"])
    save(finish([node], "kv_scatternd_assignment_f32", [data, indices, updates], [out],
                opset=16),
         "kv_scatternd_assignment_f32")


# ── RotaryEmbedding (float32) ────────────────────────────────────────────────
# The op #1078 deferred to ORT on a 12/12 losing grid. That deferral is
# withdrawn, so this fixture is the standing falsifier: if float32 RoPE is ever
# handed back to ORT's CPU EP, both assignment tests fail.
#
# com.microsoft::RotaryEmbedding takes (input, position_ids, cos_cache,
# sin_cache) with input [B, S, hidden] and caches [max_seq, head_size / 2].
def gen_rotary_assignment_f32():
    B, S, H, D = 1, 1, 32, 128
    x = vi("input", [B, S, H * D])
    pos = vi("position_ids", [B, S], TensorProto.INT64)
    cos = vi("cos_cache", [2048, D // 2])
    sin = vi("sin_cache", [2048, D // 2])
    out = vi("output", [B, S, H * D])
    node = helper.make_node(
        "RotaryEmbedding",
        ["input", "position_ids", "cos_cache", "sin_cache"],
        ["output"],
        domain=MS,
        num_heads=H,
    )
    save(finish([node], "rotary_assignment_f32", [x, pos, cos, sin], [out], ms=True),
         "rotary_assignment_f32", check=False)


# ── MultiHeadAttention ───────────────────────────────────────────────────────
# Decode-shaped SDPA, no KV cache inputs, so the graph stays single-node.
def gen_mha_assignment_f32():
    B, S, H, D = 1, 1, 32, 128
    hidden = H * D
    q = vi("query", [B, S, hidden])
    k = vi("key", [B, S, hidden])
    v = vi("value", [B, S, hidden])
    out = vi("output", [B, S, hidden])
    node = helper.make_node(
        "MultiHeadAttention", ["query", "key", "value"], ["output"],
        domain=MS, num_heads=H,
    )
    save(finish([node], "mha_assignment_f32", [q, k, v], [out], ms=True),
         "mha_assignment_f32", check=False)


# ── GroupQueryAttention ──────────────────────────────────────────────────────
# GQA with 32 query heads over 8 KV heads -- the llama3-shaped decode step.
# Inputs: q, k, v, past_key, past_value, seqlens_k, total_sequence_length,
# cos_cache, sin_cache. Outputs: output, present_key, present_value.
def gen_gqa_assignment_f32():
    B, S, H, KV, D, P = 1, 1, 32, 8, 128, 1023
    q = vi("query", [B, S, H * D])
    k = vi("key", [B, S, KV * D])
    v = vi("value", [B, S, KV * D])
    pk = vi("past_key", [B, KV, P, D])
    pv = vi("past_value", [B, KV, P, D])
    seqlens = vi("seqlens_k", [B], TensorProto.INT32)
    total = vi("total_sequence_length", [1], TensorProto.INT32)
    out = vi("output", [B, S, H * D])
    prk = vi("present_key", [B, KV, P + S, D])
    prv = vi("present_value", [B, KV, P + S, D])
    node = helper.make_node(
        "GroupQueryAttention",
        ["query", "key", "value", "past_key", "past_value", "seqlens_k",
         "total_sequence_length"],
        ["output", "present_key", "present_value"],
        domain=MS, num_heads=H, kv_num_heads=KV,
    )
    save(finish([node], "gqa_assignment_f32",
                [q, k, v, pk, pv, seqlens, total], [out, prk, prv], ms=True),
         "gqa_assignment_f32", check=False)


# ── com.microsoft::Attention (packed QKV) ────────────────────────────────────
# The signature the opset-23 ai.onnx::Attention arm deliberately does not
# cover: input is the *unprojected* activation and the fused weight carries
# q|k|v, so the output width is v_hidden rather than the input's last dim.
# That is the rule this fixture pins.
def gen_msft_attention_assignment_f32():
    B, S, H, D = 1, 8, 4, 16
    hidden = H * D
    x = vi("input", [B, S, hidden])
    w = vi("weights", [hidden, 3 * hidden])
    bias = vi("bias", [3 * hidden])
    out = vi("output", [B, S, hidden])
    node = helper.make_node(
        "Attention", ["input", "weights", "bias"], ["output"],
        domain=MS, num_heads=H,
    )
    save(finish([node], "msft_attention_assignment_f32", [x, w, bias], [out], ms=True),
         "msft_attention_assignment_f32", check=False)


# ── MoE ──────────────────────────────────────────────────────────────────────
# Mixtral-shaped routing at a small hidden size: 8 experts, top-2, SwiGLU-free
# (gelu) so the fixture stays a single node. Output shape == input shape, the
# rule that was missing from the plugin's shape table.
def gen_moe_assignment_f32():
    rows, hidden, experts, inter = 4, 32, 8, 64
    x = vi("input", [rows, hidden])
    probs = vi("router_probs", [rows, experts])
    w1 = vi("fc1_experts_weights", [experts, inter, hidden])
    b1 = vi("fc1_experts_bias", [experts, inter])
    w2 = vi("fc2_experts_weights", [experts, hidden, inter])
    b2 = vi("fc2_experts_bias", [experts, hidden])
    out = vi("output", [rows, hidden])
    node = helper.make_node(
        "MoE",
        ["input", "router_probs", "fc1_experts_weights", "fc1_experts_bias",
         "fc2_experts_weights", "fc2_experts_bias"],
        ["output"],
        domain=MS, k=2, activation_type="gelu",
    )
    save(finish([node], "moe_assignment_f32",
                [x, probs, w1, b1, w2, b2], [out], ms=True),
         "moe_assignment_f32", check=False)


def main():
    print("Generating attention / MoE / KV assignment fixtures …")
    gen_softmax_assignment_f32()
    gen_transpose_assignment_f32()
    gen_kv_concat_assignment_f32()
    gen_kv_scatternd_assignment_f32()
    gen_rotary_assignment_f32()
    gen_mha_assignment_f32()
    gen_gqa_assignment_f32()
    gen_msft_attention_assignment_f32()
    gen_moe_assignment_f32()
    print("Done.")


if __name__ == "__main__":
    main()
