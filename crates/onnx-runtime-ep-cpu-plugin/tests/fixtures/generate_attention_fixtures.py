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

# Maintained fixtures target ONNX IR version 11 and default ONNX opset 24.
# Custom-domain imports (com.microsoft) keep their own version. Enforced by the
# fixture IR/opset guard test in onnx-runtime-ep-cpu-plugin.
IR_VERSION = 11
DEFAULT_OPSET = 24


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


def finish(nodes, name, inputs, outputs, opset=DEFAULT_OPSET, ms=False, inits=None):
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits or [])
    imports = [helper.make_opsetid("", opset)]
    if ms:
        imports.append(helper.make_opsetid(MS, 1))
    model = helper.make_model(graph, opset_imports=imports)
    model.ir_version = IR_VERSION
    return model


# ── Softmax ──────────────────────────────────────────────────────────────────
# The attention softmax shape: [batch=1, heads=32, q=1, kv=1024], reduced over
# the last axis. Kept single-node so the claim cannot be inherited from a
# neighbour.
def gen_softmax_assignment_f32():
    X = vi("X", [1, 32, 1, 1024])
    Z = vi("Z", [1, 32, 1, 1024])
    node = helper.make_node("Softmax", ["X"], ["Z"], axis=-1)
    save(finish([node], "softmax_assignment_f32", [X], [Z]),
         "softmax_assignment_f32")


# ── Transpose ────────────────────────────────────────────────────────────────
# The QK layout permutation [B,S,H,D] -> [B,H,S,D], which is the one that
# actually appears between a projection and a QK GEMM.
def gen_transpose_assignment_f32():
    X = vi("X", [1, 128, 32, 128])
    Z = vi("Z", [1, 32, 128, 128])
    node = helper.make_node("Transpose", ["X"], ["Z"], perm=[0, 2, 1, 3])
    save(finish([node], "transpose_assignment_f32", [X], [Z]),
         "transpose_assignment_f32")


# ── Concat (KV cache append) ─────────────────────────────────────────────────
# past [B, H, P, D] ++ new [B, H, 1, D] along the sequence axis: the decode-step
# KV cache growth, which is where the Concat cost in this EP actually lives.
def gen_kv_concat_assignment_f32():
    past = vi("past", [1, 8, 1023, 128])
    cur = vi("cur", [1, 8, 1, 128])
    out = vi("present", [1, 8, 1024, 128])
    node = helper.make_node("Concat", ["past", "cur"], ["present"], axis=2)
    save(finish([node], "kv_concat_assignment_f32", [past, cur], [out]),
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
    save(finish([node], "kv_scatternd_assignment_f32", [data, indices, updates], [out]),
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
    gen_gqa_rotary_position_ids_assignment_f32()
    gen_msft_attention_assignment_f32()
    gen_moe_assignment_f32()
    gen_moe_assignment_f16()
    gen_qmoe_assignment_f32()
    gen_qmoe_columnwise_f32()
    gen_moe_sparse_mixer_f32()
    gen_gqa_smooth_softmax_f32()
    gen_packed_mha_assignment_f32()
    gen_trilu_assignment_f32()
    gen_scatter_elements_assignment_f32()
    print("Done.")


# ── MoE, float16 ─────────────────────────────────────────────────────────────
# Review falsifier: production mixtures are exported in half precision, and
# `moe.rs` widens f16/bf16 to f32 and narrows on the way out -- but the op
# advertised f32 alone, so every realistic MoE node was declined while the f32
# fixture passed. One dtype's worth of coverage is not coverage.
def gen_moe_assignment_f16():
    rows, hidden, experts, inter = 4, 32, 8, 64
    x = vi("input", [rows, hidden], TensorProto.FLOAT16)
    probs = vi("router_probs", [rows, experts], TensorProto.FLOAT16)
    fc1_w = helper.make_tensor(
        "fc1_w16", TensorProto.FLOAT16, [experts, inter, hidden],
        b"\x00\x3c" * (experts * inter * hidden), raw=True)
    fc2_w = helper.make_tensor(
        "fc2_w16", TensorProto.FLOAT16, [experts, hidden, inter],
        b"\x00\x3c" * (experts * hidden * inter), raw=True)
    out = vi("output", [rows, hidden], TensorProto.FLOAT16)
    node = helper.make_node(
        "MoE", ["input", "router_probs", "fc1_w16", "", "fc2_w16"], ["output"],
        domain=MS, k=2, activation_type="gelu")
    save(finish([node], "moe_assignment_f16", [x, probs], [out], ms=True,
                inits=[fc1_w, fc2_w]),
         "moe_assignment_f16", check=False)


# ── GroupQueryAttention with explicit int64 position_ids ─────────────────────
# Review falsifier for the per-slot dtype table: `position_ids` is optional
# input *9*, and leaving it off `GQA_SLOTS` made a `do_rotary` node with
# explicit positions fail the float union and go to ORT -- the same class of
# silent decline the table exists to stop, just one slot further out.
def gen_gqa_rotary_position_ids_assignment_f32():
    B, S, H, KV, D, P = 1, 1, 32, 8, 128, 1023
    q = vi("query", [B, S, H * D])
    k = vi("key", [B, S, KV * D])
    v = vi("value", [B, S, KV * D])
    pk = vi("past_key", [B, KV, P, D])
    pv = vi("past_value", [B, KV, P, D])
    seqlens = vi("seqlens_k", [B], TensorProto.INT32)
    total = vi("total_sequence_length", [1], TensorProto.INT32)
    cos = vi("cos_cache", [2048, D // 2])
    sin = vi("sin_cache", [2048, D // 2])
    pos = vi("position_ids", [B, S], TensorProto.INT64)
    out = vi("output", [B, S, H * D])
    prk = vi("present_key", [B, KV, P + S, D])
    prv = vi("present_value", [B, KV, P + S, D])
    node = helper.make_node(
        "GroupQueryAttention",
        ["query", "key", "value", "past_key", "past_value", "seqlens_k",
         "total_sequence_length", "cos_cache", "sin_cache", "position_ids"],
        ["output", "present_key", "present_value"],
        domain=MS, num_heads=H, kv_num_heads=KV, do_rotary=1,
    )
    save(finish([node], "gqa_rotary_pos_assignment_f32",
                [q, k, v, pk, pv, seqlens, total, cos, sin, pos],
                [out, prk, prv], ms=True),
         "gqa_rotary_pos_assignment_f32", check=False)


# ── QMoE ─────────────────────────────────────────────────────────────────────
# ORT does have a CPU kernel for QMoE (review loaded and ran this fixture on
# `CPUExecutionProvider` under 1.27 and 1.28), so declining it meant not running
# an op we implement rather than a load failure. int4-packed experts, so the
# weight last dim is halved and the scales carry the block structure.
def gen_qmoe_assignment_f32():
    # Blocked quantization: our kernel supports block_size >= 16 (a power of
    # two), so the scales carry one column per block rather than one per row.
    # ORT's schema default of 0 selects the column-wise form, which this kernel
    # does not implement -- a real capability gap, not a claim gap, and one ORT
    # cannot cover either since it has no CPU QMoE kernel at all.
    rows, hidden, experts, inter, block = 4, 32, 4, 64, 32
    x = vi("input", [rows, hidden])
    probs = vi("router_probs", [rows, experts])
    fc1_w = helper.make_tensor(
        "fc1_w", TensorProto.UINT8, [experts, inter, hidden // 2],
        b"\x11" * (experts * inter * hidden // 2), raw=True)
    fc1_s = helper.make_tensor(
        "fc1_s", TensorProto.FLOAT, [experts, inter, hidden // block],
        [0.01] * (experts * inter * (hidden // block)))
    fc2_w = helper.make_tensor(
        "fc2_w", TensorProto.UINT8, [experts, hidden, inter // 2],
        b"\x11" * (experts * hidden * inter // 2), raw=True)
    fc2_s = helper.make_tensor(
        "fc2_s", TensorProto.FLOAT, [experts, hidden, inter // block],
        [0.01] * (experts * hidden * (inter // block)))
    out = vi("output", [rows, hidden])
    node = helper.make_node(
        "QMoE",
        ["input", "router_probs", "fc1_w", "fc1_s", "", "fc2_w", "fc2_s"],
        ["output"],
        domain=MS, k=2, activation_type="gelu", expert_weight_bits=4,
        block_size=block,
    )
    save(finish([node], "qmoe_assignment_f32", [x, probs], [out], ms=True,
                inits=[fc1_w, fc1_s, fc2_w, fc2_s]),
         "qmoe_assignment_f32", check=False)


# ── QMoE, column-wise (the one deliberate capability decline) ────────────────
# `block_size` absent means one scale per output row, and this kernel
# implements only the blocked form. ORT's CPU kernel *does* run this, so
# claiming it and then failing in the factory would take a working model and
# kill the session -- the decline has to happen at claim time instead.
# `plugin_ort_e2e::column_wise_qmoe_is_declined_at_claim_time_not_failed_late`
# is the regression test; the fix is to implement the column-wise path.
def gen_qmoe_columnwise_f32():
    rows, hidden, experts, inter = 4, 32, 4, 64
    x = vi("input", [rows, hidden])
    probs = vi("router_probs", [rows, experts])
    fc1_w = helper.make_tensor(
        "cw_fc1_w", TensorProto.UINT8, [experts, inter, hidden // 2],
        b"\x11" * (experts * inter * hidden // 2), raw=True)
    fc1_s = helper.make_tensor(
        "cw_fc1_s", TensorProto.FLOAT, [experts, inter],
        [0.01] * (experts * inter))
    fc2_w = helper.make_tensor(
        "cw_fc2_w", TensorProto.UINT8, [experts, hidden, inter // 2],
        b"\x11" * (experts * hidden * inter // 2), raw=True)
    fc2_s = helper.make_tensor(
        "cw_fc2_s", TensorProto.FLOAT, [experts, hidden],
        [0.01] * (experts * hidden))
    out = vi("output", [rows, hidden])
    node = helper.make_node(
        "QMoE",
        ["input", "router_probs", "cw_fc1_w", "cw_fc1_s", "", "cw_fc2_w",
         "cw_fc2_s"],
        ["output"],
        domain=MS, k=2, activation_type="gelu", expert_weight_bits=4)
    save(finish([node], "qmoe_columnwise_f32", [x, probs], [out], ms=True,
                inits=[fc1_w, fc1_s, fc2_w, fc2_s]),
         "qmoe_columnwise_f32", check=False)


# ── MoE, use_sparse_mixer=1 ──────────────────────────────────────────────────
# Second review falsifier, same class as the column-wise QMoE one above. The
# sparse-mixer router (Phi-3.5-MoE / GRIN-MoE) is rejected by
# `MoeAttributes::from_node`, which runs in the kernel factory -- after ORT has
# already compiled the node onto us, where no fallback recovers. ORT's own CPU
# MoE kernel runs it, so the decline has to happen at claim time.
# `plugin_ort_e2e::factory_only_capability_limits_are_declined_at_claim_time`
# is the regression test; the fix is to implement the sparse mixer.
def gen_moe_sparse_mixer_f32():
    rows, hidden, experts, inter = 4, 32, 8, 64
    x = vi("input", [rows, hidden])
    probs = vi("router_probs", [rows, experts])
    w1 = vi("fc1_experts_weights", [experts, inter, hidden])
    w2 = vi("fc2_experts_weights", [experts, hidden, inter])
    out = vi("output", [rows, hidden])
    node = helper.make_node(
        "MoE",
        ["input", "router_probs", "fc1_experts_weights", "",
         "fc2_experts_weights"],
        ["output"],
        domain=MS, k=2, activation_type="gelu", use_sparse_mixer=1)
    save(finish([node], "moe_sparse_mixer_f32", [x, probs, w1, w2], [out], ms=True),
         "moe_sparse_mixer_f32", check=False)


# ── GroupQueryAttention, smooth_softmax=1 ────────────────────────────────────
# Same class again. `smooth_softmax = 1` (Gemma-style attention sink) is
# rejected in the GQA factory; ORT's CPU kernel implements it
# (`use_smooth_softmax_ = ... == 1`), so claiming and then failing turns a model
# that runs today into one that will not load.
def gen_gqa_smooth_softmax_f32():
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
        domain=MS, num_heads=H, kv_num_heads=KV, smooth_softmax=1,
    )
    save(finish([node], "gqa_smooth_softmax_f32",
                [q, k, v, pk, pv, seqlens, total], [out, prk, prv], ms=True),
         "gqa_smooth_softmax_f32", check=False)


# ── PackedMultiHeadAttention ─────────────────────────────────────────────────
# Rank-2 token-packed SDPA: output is [token_count, v_hidden], which is why it
# cannot use the SameAsInput(0) rule the other rescued ops share.
def gen_packed_mha_assignment_f32():
    tokens, H, D, B = 8, 4, 16, 1
    hidden = H * D
    q = vi("query", [tokens, hidden])
    k = vi("key", [tokens, hidden])
    v = vi("value", [tokens, hidden])
    token_offset = vi("token_offset", [B, tokens], TensorProto.INT32)
    cumseq = vi("cumulative_sequence_length", [B + 1], TensorProto.INT32)
    out = vi("output", [tokens, hidden])
    node = helper.make_node(
        "PackedMultiHeadAttention",
        ["query", "key", "value", "", "token_offset",
         "cumulative_sequence_length"],
        ["output"], domain=MS, num_heads=H,
    )
    save(finish([node], "packed_mha_assignment_f32",
                [q, k, v, token_offset, cumseq], [out], ms=True),
         "packed_mha_assignment_f32", check=False)


# ── Trilu ────────────────────────────────────────────────────────────────────
# The causal-mask builder. Shape-preserving, so the rule is SameAsInput(0).
def gen_trilu_assignment_f32():
    x = vi("input", [1, 1, 128, 128])
    out = vi("output", [1, 1, 128, 128])
    k = helper.make_tensor("k", TensorProto.INT64, [], [0])
    node = helper.make_node("Trilu", ["input", "k"], ["output"], upper=1)
    save(finish([node], "trilu_assignment_f32", [x], [out], inits=[k]),
         "trilu_assignment_f32")


# ── ScatterElements ──────────────────────────────────────────────────────────
# The other KV-cache write form, alongside ScatterND.
def gen_scatter_elements_assignment_f32():
    B, KV, T, D = 1, 8, 1024, 128
    data = vi("data", [B, KV, T, D])
    updates = vi("updates", [B, KV, 1, D])
    indices = helper.make_tensor(
        "indices", TensorProto.INT64, [B, KV, 1, D], [0] * (B * KV * D))
    out = vi("output", [B, KV, T, D])
    node = helper.make_node(
        "ScatterElements", ["data", "indices", "updates"], ["output"], axis=2)
    save(finish([node], "scatter_elements_assignment_f32",
                [data, updates], [out], inits=[indices]),
         "scatter_elements_assignment_f32")


if __name__ == "__main__":
    main()
