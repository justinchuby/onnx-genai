#!/usr/bin/env python3
"""Generate single-node com.microsoft::MultiHeadAttention models (the operator
#1044's vectorised `sdpa_f32` path actually serves) with fully static shapes.

SYNTHETIC DATA NOTICE: no trained weights. Only the head/head_dim/sequence
dimensions come from public model configs; tensor contents are the benchmark
harness's deterministic synthetic pattern, fed identically to both runtimes.

Slot order (com.microsoft::MultiHeadAttention):
  0 query 1 key 2 value 3 bias 4 key_padding_mask 5 attention_bias
  6 past_key 7 past_value
Outputs: 0 output 1 present_key 2 present_value
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper

OPSET = 17


def build_mha(
    path: Path,
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    q_seq: int,
    kv_seq: int,
    unidirectional: bool = False,
    past_seq: int = 0,
) -> None:
    """One MHA node. `kv_seq` counts the *new* key/value tokens; `past_seq`
    prepends a past-KV cache, so the attended length is `past_seq + kv_seq`.

    `unidirectional` is only well defined when the causal offset is
    unambiguous. ORT resolves the offset from the past-KV cache, so a graph
    that sets `unidirectional=1` while feeding one long fused K/V and no past
    input is *not* a chunked-prefill graph -- ORT's output there matches no
    causal convention at all (verified against a NumPy oracle over every offset
    in `0..=kv_seq`, #1685). Such a cell cannot be a benchmark row, because
    neither runtime is computing a defined answer, so the guard below rejects
    it rather than letting it become a silent parity failure.
    """
    if unidirectional and q_seq > 1 and past_seq == 0 and q_seq != kv_seq:
        raise ValueError(
            f"{path.name}: unidirectional with q_seq={q_seq} != kv_seq={kv_seq} and no "
            "past-KV is undefined in ORT; express chunked prefill with past_seq instead"
        )
    hidden = num_heads * head_dim
    inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch, q_seq, hidden]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch, kv_seq, hidden]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch, kv_seq, hidden]),
    ]
    node_inputs = ["query", "key", "value"]
    output_names = ["output"]
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, q_seq, hidden])
    ]
    if past_seq:
        past_shape = [batch, num_heads, past_seq, head_dim]
        present_shape = [batch, num_heads, past_seq + kv_seq, head_dim]
        inputs += [
            helper.make_tensor_value_info("past_key", TensorProto.FLOAT, past_shape),
            helper.make_tensor_value_info("past_value", TensorProto.FLOAT, past_shape),
        ]
        # Slots 3-5 (bias, key_padding_mask, attention_bias) stay empty.
        node_inputs += ["", "", "", "past_key", "past_value"]
        output_names += ["present_key", "present_value"]
        outputs += [
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, present_shape),
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT, present_shape
            ),
        ]
    node = helper.make_node(
        "MultiHeadAttention",
        node_inputs,
        output_names,
        name="mha",
        domain="com.microsoft",
        num_heads=num_heads,
        scale=float(1.0 / math.sqrt(head_dim)),
        unidirectional=int(unidirectional),
    )
    graph = helper.make_graph([node], "mha_bench", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    # (name, batch, heads, head_dim, q_seq, kv_seq, unidirectional, past_seq)
    #
    # The first block is encoder/prefill-shaped and bidirectional. The rows
    # after it exist because #1685 found the generator matrix had no decode
    # shape (`q_seq == 1`), never set `unidirectional`, and never supplied a
    # past-KV cache -- so the causal branch and the causal *offset*, the two
    # things every autoregressive model exercises on every token, were
    # benchmarked by nothing.
    cases = [
        ("bert_base_s128", 1, 12, 64, 128, 128, False, 0),
        ("bert_base_s384", 1, 12, 64, 384, 384, False, 0),
        ("bert_base_b8_s128", 8, 12, 64, 128, 128, False, 0),
        ("bert_large_s128", 1, 16, 64, 128, 128, False, 0),
        ("vit_b16_s197", 1, 12, 64, 197, 197, False, 0),
        ("clip_l14_s257", 1, 16, 64, 257, 257, False, 0),
        ("whisper_cross_s1500", 1, 12, 64, 448, 1500, False, 0),
        # Causal prefill: the first forward pass of a decoder-only model.
        # q_seq == kv_seq, so the causal offset is zero and unambiguous.
        ("llama_prefill_s128_causal", 1, 32, 128, 128, 128, True, 0),
        ("llama_prefill_s512_causal", 1, 32, 128, 512, 512, True, 0),
        ("phi35_prefill_s256_causal", 1, 32, 96, 256, 256, True, 0),
        # Decode, fused KV: one query row against an already-concatenated
        # cache. `unidirectional` is a no-op at q_seq == 1 (both runtimes
        # attend the whole cache), so this cell is well defined; it isolates
        # the GEMV shapes without the cache-concat copy.
        ("llama_decode_kv128", 1, 32, 128, 1, 128, True, 0),
        ("llama_decode_kv1024", 1, 32, 128, 1, 1024, True, 0),
        ("llama_decode_kv4096", 1, 32, 128, 1, 4096, True, 0),
        ("llama_decode_b8_kv1024", 8, 32, 128, 1, 1024, True, 0),
        ("bert_base_decode_kv1024", 1, 12, 64, 1, 1024, True, 0),
        # Decode and chunked prefill as the runtime actually emits them: new
        # tokens plus a past-KV cache, which is what makes the causal offset
        # load-bearing. `llama_chunk8_past1016` is the speculative-decode /
        # chunked-prefill shape; it replaces a fused-KV cell that asked ORT
        # for an undefined configuration and failed parity 24/24 (#1685).
        ("llama_decode_past1023", 1, 32, 128, 1, 1, True, 1023),
        ("llama_chunk8_past1016", 1, 32, 128, 8, 8, True, 1016),
        ("llama_chunk32_past992", 1, 32, 128, 32, 32, True, 992),
    ]
    for name, b, h, d, q, kv, uni, past in cases:
        path = args.out_dir / f"mha_{name}.onnx"
        build_mha(
            path,
            batch=b,
            num_heads=h,
            head_dim=d,
            q_seq=q,
            kv_seq=kv,
            unidirectional=uni,
            past_seq=past,
        )
        print(path.name)


if __name__ == "__main__":
    main()
