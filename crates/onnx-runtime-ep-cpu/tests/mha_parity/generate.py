#!/usr/bin/env python3
"""Generate ORT-1.26.0 parity goldens for `com.microsoft::MultiHeadAttention`.

For each scenario this builds a single-node ONNX graph, runs it through
onnxruntime's CPU execution provider on deterministic random inputs, and writes
the inputs + ORT reference outputs as a Rust data file
(`../mha_parity_cases.rs`) consumed by `tests/mha_ort_parity.rs`.

Re-generate after changing scenarios:

    python3 crates/onnx-runtime-ep-cpu/tests/mha_parity/generate.py

Requires: onnx, onnxruntime==1.26.0, numpy.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

assert ort.__version__ == "1.26.0", f"expected ORT 1.26.0, got {ort.__version__}"

DOMAIN = "com.microsoft"

# Input slot order of com.microsoft::MultiHeadAttention.
SLOTS = [
    "query",
    "key",
    "value",
    "bias",
    "key_padding_mask",
    "attention_bias",
    "past_key",
    "past_value",
]


def build_model(inputs, attrs, num_outputs):
    """Single MHA node graph. `inputs` maps slot name -> numpy array (present
    slots only). Omitted optional slots become empty-string node inputs."""
    max_slot = max(SLOTS.index(name) for name in inputs)
    node_inputs = []
    graph_inputs = []
    for slot in range(max_slot + 1):
        name = SLOTS[slot]
        if name in inputs:
            arr = inputs[name]
            elem = (
                TensorProto.INT32
                if arr.dtype == np.int32
                else TensorProto.FLOAT
            )
            node_inputs.append(name)
            graph_inputs.append(helper.make_tensor_value_info(name, elem, arr.shape))
        else:
            node_inputs.append("")

    out_names = ["output", "present_key", "present_value"][:num_outputs]
    graph_outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in out_names
    ]

    node = helper.make_node(
        "MultiHeadAttention",
        node_inputs,
        out_names,
        domain=DOMAIN,
        **attrs,
    )
    graph = helper.make_graph([node], "mha", graph_inputs, graph_outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid(DOMAIN, 1),
        ],
        ir_version=10,
    )
    return model, out_names


def run(inputs, attrs, num_outputs):
    model, out_names = build_model(inputs, attrs, num_outputs)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {name: arr for name, arr in inputs.items()}
    outs = sess.run(out_names, feeds)
    return out_names, outs


def rnd(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def make_cases():
    cases = []

    def add(name, inputs, attrs, num_outputs):
        out_names, outs = run(inputs, attrs, num_outputs)
        cases.append(
            {
                "name": name,
                "attrs": attrs,
                "inputs": inputs,
                "outputs": list(zip(out_names, outs)),
            }
        )
        diffs = ", ".join(f"{n}{o.shape}" for n, o in zip(out_names, outs))
        print(f"  [{name}] outputs: {diffs}")

    # (a) basic self-attention, no mask; request present_key/value too.
    rng = np.random.default_rng(1)
    B, S, N, H = 2, 4, 3, 5
    D = N * H
    add(
        "self_attn_basic",
        {"query": rnd(rng, B, S, D), "key": rnd(rng, B, S, D), "value": rnd(rng, B, S, D)},
        {"num_heads": N},
        3,
    )

    # (b) causal / unidirectional self-attention.
    rng = np.random.default_rng(2)
    B, S, N, H = 1, 5, 2, 4
    D = N * H
    add(
        "self_attn_causal",
        {"query": rnd(rng, B, S, D), "key": rnd(rng, B, S, D), "value": rnd(rng, B, S, D)},
        {"num_heads": N, "unidirectional": 1},
        3,
    )

    # (c) cross-attention with kv_seq != q_seq (rank-3 BSNH).
    rng = np.random.default_rng(3)
    B, Sq, L, N, H = 2, 3, 6, 2, 4
    D = N * H
    add(
        "cross_attn_bsnh",
        {"query": rnd(rng, B, Sq, D), "key": rnd(rng, B, L, D), "value": rnd(rng, B, L, D)},
        {"num_heads": N},
        1,
    )

    # (d) cross-attention with already-transposed rank-4 K/V (BNSH).
    rng = np.random.default_rng(4)
    B, Sq, L, N, H = 1, 3, 5, 2, 4
    D = N * H
    add(
        "cross_attn_bnsh_rank4",
        {
            "query": rnd(rng, B, Sq, D),
            "key": rnd(rng, B, N, L, H),
            "value": rnd(rng, B, N, L, H),
        },
        {"num_heads": N},
        3,
    )

    # (e) incremental decode with past KV cache (S == 1, present concat).
    rng = np.random.default_rng(5)
    B, N, H, P = 1, 2, 4, 4
    D = N * H
    add(
        "past_kv_decode",
        {
            "query": rnd(rng, B, 1, D),
            "key": rnd(rng, B, 1, D),
            "value": rnd(rng, B, 1, D),
            "past_key": rnd(rng, B, N, P, H),
            "past_value": rnd(rng, B, N, P, H),
        },
        {"num_heads": N, "unidirectional": 1},
        3,
    )

    # (f) differing V head size (H=4, H_v=3).
    rng = np.random.default_rng(6)
    B, S, N, H, Hv = 1, 3, 2, 4, 3
    D, Dv = N * H, N * Hv
    add(
        "diff_v_head_size",
        {"query": rnd(rng, B, S, D), "key": rnd(rng, B, S, D), "value": rnd(rng, B, S, Dv)},
        {"num_heads": N},
        1,
    )

    # (g) Q/K/V bias (length D + D + D_v).
    rng = np.random.default_rng(7)
    B, S, N, H = 1, 3, 2, 4
    D = N * H
    add(
        "with_bias",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, S, D),
            "value": rnd(rng, B, S, D),
            "bias": rnd(rng, 3 * D),
        },
        {"num_heads": N},
        1,
    )

    # (h) additive attention_bias (B, N, S, T).
    rng = np.random.default_rng(8)
    B, S, L, N, H = 2, 3, 3, 2, 4
    D = N * H
    add(
        "with_attn_bias_bnst",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, L, D),
            "value": rnd(rng, B, L, D),
            "attention_bias": rnd(rng, B, N, S, L),
        },
        {"num_heads": N},
        1,
    )

    # (i) broadcast attention_bias (1, 1, S, T) + explicit scale override.
    rng = np.random.default_rng(9)
    B, S, L, N, H = 1, 4, 4, 2, 4
    D = N * H
    add(
        "attn_bias_broadcast_scale",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, L, D),
            "value": rnd(rng, B, L, D),
            "attention_bias": rnd(rng, 1, 1, S, L),
        },
        {"num_heads": N, "scale": 0.3},
        1,
    )

    # (j) 2D key_padding_mask (B, T) int32.
    rng = np.random.default_rng(10)
    B, S, L, N, H = 2, 3, 4, 2, 4
    D = N * H
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int32)
    add(
        "key_padding_mask_2d",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, L, D),
            "value": rnd(rng, B, L, D),
            "key_padding_mask": mask,
        },
        {"num_heads": N},
        1,
    )

    # (k) 1D key length mask (B,) int32.
    rng = np.random.default_rng(11)
    B, S, L, N, H = 2, 3, 4, 2, 4
    D = N * H
    keylen = np.array([4, 2], dtype=np.int32)
    add(
        "key_len_mask_1d",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, L, D),
            "value": rnd(rng, B, L, D),
            "key_padding_mask": keylen,
        },
        {"num_heads": N},
        1,
    )

    # (l) 3D key_padding_mask (B, S, T) int32.
    rng = np.random.default_rng(12)
    B, S, L, N, H = 1, 3, 3, 2, 4
    D = N * H
    mask3d = np.array([[[1, 1, 0], [1, 1, 1], [1, 0, 0]]], dtype=np.int32)
    add(
        "key_padding_mask_3d",
        {
            "query": rnd(rng, B, S, D),
            "key": rnd(rng, B, L, D),
            "value": rnd(rng, B, L, D),
            "key_padding_mask": mask3d,
        },
        {"num_heads": N},
        1,
    )

    return cases


def fmt_f32(x):
    r = repr(float(x))
    if "inf" in r or "nan" in r:
        raise ValueError(f"non-finite golden value {r}")
    if "." not in r and "e" not in r and "E" not in r:
        r += ".0"
    return r + "f32"


def emit(cases, path):
    lines = []
    lines.append("// @generated by tests/mha_parity/generate.py — DO NOT EDIT.")
    lines.append("// ORT 1.26.0 parity goldens for com.microsoft::MultiHeadAttention.")
    lines.append("#[rustfmt::skip]")
    lines.append("#[allow(clippy::all, clippy::pedantic)]")
    lines.append("fn cases() -> Vec<Case> {")
    lines.append("    vec![")
    for c in cases:
        attrs = c["attrs"]
        num_heads = attrs["num_heads"]
        scale = attrs.get("scale")
        unidir = attrs.get("unidirectional", 0)
        mfv = attrs.get("mask_filter_value")
        lines.append("        Case {")
        lines.append(f'            name: "{c["name"]}",')
        lines.append(f"            num_heads: {num_heads},")
        lines.append(
            f"            scale: {'Some(' + fmt_f32(scale) + ')' if scale is not None else 'None'},"
        )
        lines.append(
            f"            mask_filter_value: {'Some(' + fmt_f32(mfv) + ')' if mfv is not None else 'None'},"
        )
        lines.append(f"            unidirectional: {unidir},")
        # inputs (present slots only)
        lines.append("            inputs: vec![")
        for name, arr in c["inputs"].items():
            slot = SLOTS.index(name)
            shape = ", ".join(str(d) for d in arr.shape)
            flat = arr.reshape(-1)
            if arr.dtype == np.int32:
                data = ", ".join(str(int(v)) for v in flat)
                val = f"InputData::I32(vec![{data}])"
            else:
                data = ", ".join(fmt_f32(v) for v in flat)
                val = f"InputData::F32(vec![{data}])"
            lines.append(
                f"                CaseInput {{ slot: {slot}, shape: vec![{shape}], data: {val} }},"
            )
        lines.append("            ],")
        # outputs
        lines.append("            outputs: vec![")
        for oname, arr in c["outputs"]:
            arr = np.asarray(arr)
            shape = ", ".join(str(d) for d in arr.shape)
            data = ", ".join(fmt_f32(v) for v in arr.reshape(-1))
            lines.append(
                f"                CaseOutput {{ shape: vec![{shape}], data: vec![{data}] }},"
            )
        lines.append("            ],")
        lines.append("        },")
    lines.append("    ]")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "cases.rs")
    print("generating MHA parity goldens with ORT", ort.__version__)
    cases = make_cases()
    emit(cases, os.path.normpath(out))
