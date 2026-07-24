#!/usr/bin/env python3
"""Regenerate ORT-1.26 Qwen3.5 hybrid-op parity goldens.

The checked-in f32 input bit patterns are stable test scenarios. This script
loads them from cases.rs, runs single-node models through ORT's CPU EP, and
rewrites the inputs plus fresh ORT reference outputs.

    python3 crates/onnx-runtime-ep-cpu/tests/qwen35_parity/generate.py

Requires: onnx==1.22, onnxruntime==1.26.0, numpy.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

assert ort.__version__ == "1.26.0", f"expected ORT 1.26.0, got {ort.__version__}"

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "cases.rs"


def constants(source):
    arrays = {}
    for name, body in re.findall(
        r"pub const (\w+): \[[^\]]+\] = \[(.*?)\];", source, re.DOTALL
    ):
        arrays[name] = [int(value, 0) for value in re.findall(r"0x[0-9a-f]+|\d+", body)]
    scalars = {
        name: value
        for name, value in re.findall(
            r"pub const (\w+): (?:bool|f32) = ([^;]+);", source
        )
    }
    return arrays, scalars


def f32(bits, shape):
    return np.asarray(bits, dtype=np.uint32).view(np.float32).reshape(shape)


def bits(values):
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()


def rust_arr(name, values):
    values = bits(values)
    items = ", ".join(f"0x{value:08x}" for value in values)
    return f"    pub const {name}: [u32; {len(values)}] = [{items}];\n"


def session(node, inputs, outputs, name):
    graph = helper.make_graph([node], name, inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def conv_session(channels, kernel, sequence, batch, activation):
    inputs = [
        helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, [batch, channels, sequence]
        ),
        helper.make_tensor_value_info(
            "w", TensorProto.FLOAT, [channels, 1, kernel]
        ),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [channels]),
        helper.make_tensor_value_info(
            "st", TensorProto.FLOAT, [batch, channels, kernel - 1]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, [batch, channels, sequence]
        ),
        helper.make_tensor_value_info(
            "s2", TensorProto.FLOAT, [batch, channels, kernel - 1]
        ),
    ]
    node = helper.make_node(
        "CausalConvWithState",
        ["x", "w", "b", "st"],
        ["y", "s2"],
        domain="com.microsoft",
        ndim=1,
        activation=activation,
    )
    return session(node, inputs, outputs, "causal_conv")


def linear_attention_session(heads, d_k, d_v, sequence, batch, scale):
    inputs = [
        helper.make_tensor_value_info(
            "q", TensorProto.FLOAT, [batch, sequence, heads * d_k]
        ),
        helper.make_tensor_value_info(
            "k", TensorProto.FLOAT, [batch, sequence, heads * d_k]
        ),
        helper.make_tensor_value_info(
            "v", TensorProto.FLOAT, [batch, sequence, heads * d_v]
        ),
        helper.make_tensor_value_info(
            "st", TensorProto.FLOAT, [batch, heads, d_k, d_v]
        ),
        helper.make_tensor_value_info(
            "g", TensorProto.FLOAT, [batch, sequence, heads]
        ),
        helper.make_tensor_value_info(
            "be", TensorProto.FLOAT, [batch, sequence, heads]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "o", TensorProto.FLOAT, [batch, sequence, heads * d_v]
        ),
        helper.make_tensor_value_info(
            "s2", TensorProto.FLOAT, [batch, heads, d_k, d_v]
        ),
    ]
    node = helper.make_node(
        "LinearAttention",
        ["q", "k", "v", "st", "g", "be"],
        ["o", "s2"],
        domain="com.microsoft",
        q_num_heads=heads,
        kv_num_heads=heads,
        update_rule="gated_delta",
        scale=float(scale),
    )
    return session(node, inputs, outputs, "linear_attention")


def main():
    arrays, scalars = constants(OUTPUT.read_text())
    lines = [
        "// Auto-generated ORT-1.26 golden fixtures for the Qwen3.5 hybrid linear-attention\n",
        "// kernels (CausalConvWithState, LinearAttention). Values are bit-exact f32\n",
        "// (`f32::from_bits`). Regenerate with tests/qwen35_parity/generate.py. DO NOT EDIT.\n",
        "#[allow(clippy::all)]\n",
        "pub mod conv {\n",
        "    /// (name, C, K, S, B, activation_is_silu)\n",
    ]

    for name in ("CONVA", "CONVB", "CONVC"):
        batch, channels, kernel, sequence = arrays[f"{name}_DIMS"]
        silu = scalars[f"{name}_SILU"] == "true"
        activation = "silu" if silu else "none"
        x = f32(arrays[f"{name}_X"], (batch, channels, sequence))
        weight = f32(arrays[f"{name}_W"], (channels, 1, kernel))
        bias = f32(arrays[f"{name}_B"], (channels,))
        state = f32(arrays[f"{name}_STATE"], (batch, channels, kernel - 1))
        output, present = conv_session(
            channels, kernel, sequence, batch, activation
        ).run(None, {"x": x, "w": weight, "b": bias, "st": state})
        lines += [
            f"    // case {name}: C={channels} K={kernel} S={sequence} B={batch} act={activation}\n",
            f"    pub const {name}_DIMS: [usize; 4] = [{batch}, {channels}, {kernel}, {sequence}];\n",
            f"    pub const {name}_SILU: bool = {str(silu).lower()};\n",
            rust_arr(f"{name}_X", x),
            rust_arr(f"{name}_W", weight),
            rust_arr(f"{name}_B", bias),
            rust_arr(f"{name}_STATE", state),
            rust_arr(f"{name}_Y", output),
            rust_arr(f"{name}_PRESENT", present),
        ]
    lines += ["}\n\n", "#[allow(clippy::all)]\n", "pub mod la {\n"]

    for name in ("LAA", "LAB", "LAC"):
        batch, heads, d_k, d_v, sequence = arrays[f"{name}_DIMS"]
        scale = float(scalars[f"{name}_SCALE"].removesuffix("f32"))
        q = f32(arrays[f"{name}_Q"], (batch, sequence, heads * d_k))
        k = f32(arrays[f"{name}_K"], (batch, sequence, heads * d_k))
        v = f32(arrays[f"{name}_V"], (batch, sequence, heads * d_v))
        state = f32(arrays[f"{name}_STATE"], (batch, heads, d_k, d_v))
        decay = f32(arrays[f"{name}_G"], (batch, sequence, heads))
        beta = f32(arrays[f"{name}_BETA"], (batch, sequence, heads))
        output, present = linear_attention_session(
            heads, d_k, d_v, sequence, batch, scale
        ).run(
            None,
            {"q": q, "k": k, "v": v, "st": state, "g": decay, "be": beta},
        )
        lines += [
            f"    // case {name}: H={heads} Dk={d_k} Dv={d_v} S={sequence} B={batch} scale={scale}\n",
            f"    pub const {name}_DIMS: [usize; 5] = [{batch}, {heads}, {d_k}, {d_v}, {sequence}];\n",
            f"    pub const {name}_SCALE: f32 = {scale}f32;\n",
            rust_arr(f"{name}_Q", q),
            rust_arr(f"{name}_K", k),
            rust_arr(f"{name}_V", v),
            rust_arr(f"{name}_STATE", state),
            rust_arr(f"{name}_G", decay),
            rust_arr(f"{name}_BETA", beta),
            rust_arr(f"{name}_O", output),
            rust_arr(f"{name}_PRESENT", present),
        ]
    lines.append("}\n")

    OUTPUT.write_text("".join(lines))
    subprocess.run(["rustfmt", "--edition", "2024", str(OUTPUT)], check=True)
    print(f"wrote {OUTPUT} with ORT {ort.__version__}")


if __name__ == "__main__":
    main()
