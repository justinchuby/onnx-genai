#!/usr/bin/env python3
"""Isolated single-node graphs for the transforms that *surround* attention:
Softmax, RotaryEmbedding, and the KV-cache append/gather copies.

The point is requirement 4: prove that #1044/#1052's attention wins are not
just time pushed into a neighbouring node, and measure the softmax / RoPE /
KV-copy paths against ORT on their own terms.

SYNTHETIC DATA NOTICE: no trained weights. Only the head / head_dim / sequence
dimensions come from public model configs; tensor contents are the benchmark
harness's deterministic synthetic pattern, fed identically to both runtimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OPSET = 17


def _save(path: Path, graph: onnx.GraphProto, *, ms_domain: bool) -> None:
    opsets = [helper.make_opsetid("", OPSET)]
    if ms_domain:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 10
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    print(f"wrote {path}")


def build_softmax(path: Path, *, rows: int, cols: int) -> None:
    """Attention-logit shaped softmax: `rows` = batch*heads*q_seq."""
    node = helper.make_node("Softmax", ["x"], ["y"], axis=-1)
    graph = helper.make_graph(
        [node],
        "softmax",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [rows, cols])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [rows, cols])],
    )
    _save(path, graph, ms_domain=False)


def build_rotary(
    path: Path,
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    seq: int,
    max_seq: int = 4096,
    interleaved: bool = False,
) -> None:
    """com.microsoft::RotaryEmbedding on BSNH input (3D hidden layout)."""
    hidden = num_heads * head_dim
    half = head_dim // 2
    node = helper.make_node(
        "RotaryEmbedding",
        ["input", "position_ids", "cos_cache", "sin_cache"],
        ["output"],
        domain="com.microsoft",
        interleaved=1 if interleaved else 0,
        num_heads=num_heads,
    )
    graph = helper.make_graph(
        [node],
        "rotary",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [batch, seq, hidden]
            ),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [batch, seq]),
            helper.make_tensor_value_info(
                "cos_cache", TensorProto.FLOAT, [max_seq, half]
            ),
            helper.make_tensor_value_info(
                "sin_cache", TensorProto.FLOAT, [max_seq, half]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [batch, seq, hidden]
            )
        ],
    )
    _save(path, graph, ms_domain=True)


def build_kv_append(
    path: Path,
    *,
    batch: int,
    kv_heads: int,
    head_dim: int,
    past: int,
    new: int,
) -> None:
    """The KV-cache append copy on its own: concat past with the new step.

    This is exactly the traffic `fill_present` performs inside GQA, expressed
    as the graph-level ops a non-fused model would use, so the two are
    comparable and neither runtime can hide the copy in a fused kernel.
    """
    nodes = [
        helper.make_node("Concat", ["past_key", "new_key"], ["present_key"], axis=2),
        helper.make_node("Concat", ["past_value", "new_value"], ["present_value"], axis=2),
    ]
    shape_past = [batch, kv_heads, past, head_dim]
    shape_new = [batch, kv_heads, new, head_dim]
    shape_out = [batch, kv_heads, past + new, head_dim]
    graph = helper.make_graph(
        nodes,
        "kv_append",
        [
            helper.make_tensor_value_info("past_key", TensorProto.FLOAT, shape_past),
            helper.make_tensor_value_info("new_key", TensorProto.FLOAT, shape_new),
            helper.make_tensor_value_info("past_value", TensorProto.FLOAT, shape_past),
            helper.make_tensor_value_info("new_value", TensorProto.FLOAT, shape_new),
        ],
        [
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, shape_out),
            helper.make_tensor_value_info("present_value", TensorProto.FLOAT, shape_out),
        ],
    )
    _save(path, graph, ms_domain=False)


def build_transpose(
    path: Path, *, batch: int, seq: int, num_heads: int, head_dim: int
) -> None:
    """BSNH -> BNSH, the transform #1052 blocked and parallelised."""
    nodes = [
        helper.make_node(
            "Reshape", ["input", "shape4"], ["r"]
        ),
        helper.make_node("Transpose", ["r"], ["output"], perm=[0, 2, 1, 3]),
    ]
    hidden = num_heads * head_dim
    shape4 = helper.make_tensor(
        "shape4", TensorProto.INT64, [4], [batch, seq, num_heads, head_dim]
    )
    graph = helper.make_graph(
        nodes,
        "transpose_bsnh",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, seq, hidden])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [batch, num_heads, seq, head_dim]
            )
        ],
        initializer=[shape4],
    )
    _save(path, graph, ms_domain=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out

    # --- Softmax at real attention-logit shapes -------------------------
    # llama3-8B decode: 32 q heads x 1 query row, kv_seq = past+1.
    for kv in (1024, 2048, 4096, 8192):
        build_softmax(out / f"sm_decode_h32_kv{kv}.onnx", rows=32, cols=kv)
    # BERT-base b8 s128: 8*12*128 = 12288 rows of 128.
    build_softmax(out / "sm_bert_b8_s128.onnx", rows=12288, cols=128)
    # Whisper cross-attention b1: 20 heads * 1500 rows of 1500.
    build_softmax(out / "sm_whisper_cross.onnx", rows=20 * 1500, cols=1500)
    # Prefill llama3 b1 s512: 32 heads * 512 rows of 512.
    build_softmax(out / "sm_prefill_h32_s512.onnx", rows=32 * 512, cols=512)

    # --- RoPE ----------------------------------------------------------
    for seq in (1, 128, 512):
        build_rotary(
            out / f"rope_llama3_s{seq}.onnx",
            batch=1,
            num_heads=32,
            head_dim=128,
            seq=seq,
        )
    build_rotary(
        out / "rope_llama3_b8_s1.onnx", batch=8, num_heads=32, head_dim=128, seq=1
    )
    # GPT-J convention (interleaved=1) rotates adjacent even/odd channels
    # instead of the two halves, which is a different inner loop with a
    # different vectorisation. Cover it at decode and prefill lengths.
    for seq in (1, 128, 512):
        build_rotary(
            out / f"rope_gptj_il_s{seq}.onnx",
            batch=1,
            num_heads=32,
            head_dim=128,
            seq=seq,
            interleaved=True,
        )

    # --- KV-cache append copies ----------------------------------------
    for past in (1023, 2047, 4095, 8191):
        build_kv_append(
            out / f"kvcat_llama3_p{past}.onnx",
            batch=1,
            kv_heads=8,
            head_dim=128,
            past=past,
            new=1,
        )
    build_kv_append(
        out / "kvcat_llama3_b8_p2047.onnx",
        batch=8,
        kv_heads=8,
        head_dim=128,
        past=2047,
        new=1,
    )

    # --- BSNH <-> BNSH transposes --------------------------------------
    build_transpose(
        out / "tr_bert_b8_s128.onnx", batch=8, seq=128, num_heads=12, head_dim=64
    )
    build_transpose(
        out / "tr_whisper_s1500.onnx", batch=1, seq=1500, num_heads=20, head_dim=64
    )
    build_transpose(
        out / "tr_llama3_s512.onnx", batch=1, seq=512, num_heads=32, head_dim=128
    )


if __name__ == "__main__":
    main()
