#!/usr/bin/env python3
"""Generate ORT-1.26.0 parity goldens for `com.microsoft::Attention` (packed QKV).

For each scenario this builds a single-node ONNX graph, runs it through
onnxruntime's CPU execution provider on deterministic random inputs, and writes
the inputs + ORT reference outputs as a Rust data file (`./cases.rs`) consumed
by `tests/msft_attention_ort_parity.rs`.

Re-generate after changing scenarios:

    python3 crates/onnx-runtime-ep-cpu/tests/msft_attention_parity/generate.py

Requires: onnx, onnxruntime==1.26.0, numpy.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

assert ort.__version__ == "1.26.0", f"expected ORT 1.26.0, got {ort.__version__}"

DOMAIN = "com.microsoft"

# Input slot order of com.microsoft::Attention.
SLOTS = [
    "input",
    "weights",
    "bias",
    "mask_index",
    "past",
    "attention_bias",
]


def build_model(inputs, attrs, num_outputs):
    """Single Attention-node graph. `inputs` maps slot name -> numpy array
    (present slots only). Omitted optional slots become empty-string node
    inputs."""
    max_slot = max(SLOTS.index(name) for name in inputs)
    node_inputs = []
    graph_inputs = []
    for slot in range(max_slot + 1):
        name = SLOTS[slot]
        if name in inputs:
            arr = inputs[name]
            elem = TensorProto.INT32 if arr.dtype == np.int32 else TensorProto.FLOAT
            node_inputs.append(name)
            graph_inputs.append(helper.make_tensor_value_info(name, elem, arr.shape))
        else:
            node_inputs.append("")

    out_names = ["output", "present"][:num_outputs]
    graph_outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in out_names
    ]

    node = helper.make_node(
        "Attention",
        node_inputs,
        out_names,
        domain=DOMAIN,
        **attrs,
    )
    graph = helper.make_graph([node], "attn", graph_inputs, graph_outputs)
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

    # (a) basic packed-QKV self-attention (bias required by ORT CPU kernel).
    rng = np.random.default_rng(1)
    B, S, N, H, Di = 2, 4, 3, 5, 7
    D = N * H
    add(
        "self_attn_basic",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
        },
        {"num_heads": N},
        2,
    )

    # (b) unidirectional (causal) self-attention.
    rng = np.random.default_rng(2)
    B, S, N, H, Di = 1, 5, 2, 4, 6
    D = N * H
    add(
        "self_attn_causal",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
        },
        {"num_heads": N, "unidirectional": 1},
        1,
    )

    # (c) explicit scale override.
    rng = np.random.default_rng(3)
    B, S, N, H, Di = 2, 3, 2, 4, 5
    D = N * H
    add(
        "explicit_scale",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
        },
        {"num_heads": N, "scale": 0.3},
        1,
    )

    # (d) raw 2D key_padding mask (B, T) int32.
    rng = np.random.default_rng(4)
    B, S, N, H, Di = 2, 3, 2, 4, 5
    D = N * H
    mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32)
    add(
        "mask_2d_raw",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "mask_index": mask,
        },
        {"num_heads": N},
        1,
    )

    # (e) 1D key-length mask (B,) int32 (right padding).
    rng = np.random.default_rng(5)
    B, S, N, H, Di = 2, 4, 2, 4, 6
    D = N * H
    keylen = np.array([4, 2], dtype=np.int32)
    add(
        "mask_1d_keylen",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "mask_index": keylen,
        },
        {"num_heads": N},
        1,
    )

    # (f) 3D key_padding mask (B, S, T) int32.
    rng = np.random.default_rng(6)
    B, S, N, H, Di = 1, 3, 2, 4, 5
    D = N * H
    mask3d = np.array([[[1, 1, 0], [1, 1, 1], [1, 0, 0]]], dtype=np.int32)
    add(
        "mask_3d",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "mask_index": mask3d,
        },
        {"num_heads": N},
        1,
    )

    # (g) additive attention_bias (B, N, S, T).
    rng = np.random.default_rng(7)
    B, S, N, H, Di = 2, 3, 2, 4, 5
    D = N * H
    add(
        "attn_bias_bnst",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "attention_bias": rnd(rng, B, N, S, S),
        },
        {"num_heads": N},
        1,
    )

    # (h) past/present KV cache — incremental decode step (S == 1).
    rng = np.random.default_rng(8)
    B, N, H, P, Di = 1, 2, 4, 3, 6
    D = N * H
    add(
        "past_kv_decode",
        {
            "input": rnd(rng, B, 1, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "past": rnd(rng, 2, B, N, P, H),
        },
        {"num_heads": N, "unidirectional": 1},
        2,
    )

    # (i) past/present KV cache with a multi-token query (S > 1, causal).
    rng = np.random.default_rng(9)
    B, S, N, H, P, Di = 1, 3, 2, 4, 2, 6
    D = N * H
    add(
        "past_kv_prefill",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, 3 * D),
            "bias": rnd(rng, 3 * D),
            "past": rnd(rng, 2, B, N, P, H),
        },
        {"num_heads": N, "unidirectional": 1},
        2,
    )

    # (j) asymmetric qkv_hidden_sizes (v_hidden != q_hidden).
    rng = np.random.default_rng(10)
    B, S, N, Di = 1, 3, 2, 5
    Hq, Hv = 4, 3
    Dq, Dv = N * Hq, N * Hv
    add(
        "qkv_hidden_sizes_diff_v",
        {
            "input": rnd(rng, B, S, Di),
            "weights": rnd(rng, Di, Dq + Dq + Dv),
            "bias": rnd(rng, Dq + Dq + Dv),
        },
        {"num_heads": N, "qkv_hidden_sizes": [Dq, Dq, Dv]},
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
    lines.append("// @generated by tests/msft_attention_parity/generate.py — DO NOT EDIT.")
    lines.append("// ORT 1.26.0 parity goldens for com.microsoft::Attention (packed QKV).")
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
        qkv = attrs.get("qkv_hidden_sizes")
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
        if qkv is not None:
            qkv_str = ", ".join(str(int(v)) for v in qkv)
            lines.append(f"            qkv_hidden_sizes: Some(vec![{qkv_str}]),")
        else:
            lines.append("            qkv_hidden_sizes: None,")
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
    print("generating Attention parity goldens with ORT", ort.__version__)
    cases = make_cases()
    emit(cases, os.path.normpath(out))
