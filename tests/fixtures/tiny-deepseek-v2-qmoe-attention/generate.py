#!/usr/bin/env python3
"""Generate a deterministic tiny DeepSeek-V2-style standard Attention + QMoE fixture.

The real DeepSeek-V2-Lite int4 export currently lowers MLA into standard
ai.onnx::RotaryEmbedding + ai.onnx::Attention plus sparse com.microsoft::QMoE.
This fixture intentionally locks that native path rather than a model-specific
custom MLA op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from google.protobuf import text_format
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

SEED = 7
VOCAB = 16
HIDDEN = 16
HEADS = 2
HEAD_DIM = HIDDEN // HEADS
EXPERTS = 4
TOP_K = 2
INTER = 16
BLOCK_SIZE = 16
BITS = 4
MAX_SEQUENCE = 16
PROMPT_IDS = [3]
# Filled from the generated fixture's native CPU greedy stream and locked by tests.
EXPECTED_TOKENS = [11, 11, 11, 11, 11, 11, 11, 11]


def f32(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def i64(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array.astype(np.int64), name=name)


def quantize(experts: int, out_features: int, in_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pack_size = 8 // BITS
    assert in_features % BLOCK_SIZE == 0
    packed_in = in_features // pack_size
    blocks = in_features // BLOCK_SIZE
    packed = np.zeros((experts, out_features, packed_in), dtype=np.uint8)
    scales = np.zeros((experts, out_features, blocks), dtype=np.float32)
    zero = 1 << (BITS - 1)
    mask = (1 << BITS) - 1
    for expert in range(experts):
        for output in range(out_features):
            for block in range(blocks):
                scales[expert, output, block] = 0.03 + 0.01 * ((seed + expert * 3 + output * 5 + block * 7) % 5)
                for within in range(BLOCK_SIZE):
                    depth = block * BLOCK_SIZE + within
                    centered = ((seed + expert * 11 + output * 13 + depth * 17) % 7) - 3
                    q = max(0, min(mask, centered + zero))
                    packed[expert, output, depth // pack_size] |= np.uint8(q << ((depth % pack_size) * BITS))
    return packed, scales


def write_tokenizer(path: Path) -> None:
    vocab = {str(index): index for index in range(VOCAB)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def build(output: Path) -> None:
    rng = np.random.default_rng(SEED)
    initializers: list[onnx.TensorProto] = []

    embedding = rng.normal(0.0, 0.20, size=(VOCAB, HIDDEN)).astype(np.float32)
    wq = rng.normal(0.0, 0.15, size=(HIDDEN, HIDDEN)).astype(np.float32)
    wk = rng.normal(0.0, 0.15, size=(HIDDEN, HIDDEN)).astype(np.float32)
    wv = rng.normal(0.0, 0.15, size=(HIDDEN, HIDDEN)).astype(np.float32)
    # Bias token 11 enough to make the fixture stable while still flowing through
    # Attention and QMoE for every generated step.
    lm_head = rng.normal(0.0, 0.08, size=(HIDDEN, VOCAB)).astype(np.float32)
    lm_bias = np.linspace(-0.25, 0.25, VOCAB, dtype=np.float32)
    lm_bias[11] = 1.5
    router = np.array([[3.0, 2.0, -1.0, -2.0]], dtype=np.float32)
    positions = np.arange(MAX_SEQUENCE, dtype=np.float32)[:, None]
    freqs = np.arange(HEAD_DIM // 2, dtype=np.float32)[None, :] + 1.0
    angles = positions / (10.0 * freqs)
    cos_cache = np.cos(angles).astype(np.float32)
    sin_cache = np.sin(angles).astype(np.float32)
    fc1_packed, fc1_scales = quantize(EXPERTS, INTER * 2, HIDDEN, 1)
    fc2_packed, fc2_scales = quantize(EXPERTS, HIDDEN, INTER, 2)

    for tensor in [
        f32("embedding", embedding),
        f32("wq", wq),
        f32("wk", wk),
        f32("wv", wv),
        f32("lm_head", lm_head),
        f32("lm_bias", lm_bias),
        f32("router_probs", router),
        f32("cos_cache", cos_cache),
        f32("sin_cache", sin_cache),
        numpy_helper.from_array(fc1_packed, name="fc1_experts_weights"),
        f32("fc1_scales", fc1_scales),
        numpy_helper.from_array(fc2_packed, name="fc2_experts_weights"),
        f32("fc2_scales", fc2_scales),
    ]:
        initializers.append(tensor)

    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["x"], name="token_embedding", axis=0),
        helper.make_node("MatMul", ["x", "wq"], ["q_pre"], name="q_proj"),
        helper.make_node("MatMul", ["x", "wk"], ["k_pre"], name="k_proj"),
        helper.make_node("MatMul", ["x", "wv"], ["v"], name="v_proj"),
        helper.make_node(
            "RotaryEmbedding",
            ["q_pre", "cos_cache", "sin_cache", "position_ids"],
            ["q"],
            name="q_rope",
            num_heads=HEADS,
            interleaved=1,
            rotary_embedding_dim=0,
        ),
        helper.make_node(
            "RotaryEmbedding",
            ["k_pre", "cos_cache", "sin_cache", "position_ids"],
            ["k"],
            name="k_rope",
            num_heads=HEADS,
            interleaved=1,
            rotary_embedding_dim=0,
        ),
        helper.make_node("Cast", ["attention_mask"], ["attn_mask_bool"], name="attention_mask_to_bool", to=TensorProto.BOOL),
        helper.make_node(
            "Attention",
            ["q", "k", "v", "attn_mask_bool", "past_key_values.0.key", "past_key_values.0.value"],
            ["attn", "present.0.key", "present.0.value"],
            name="standard_attention",
            q_num_heads=HEADS,
            kv_num_heads=HEADS,
            scale=float(1.0 / np.sqrt(HEAD_DIM)),
        ),
        helper.make_node(
            "QMoE",
            [
                "attn",
                "router_probs",
                "fc1_experts_weights",
                "fc1_scales",
                "",
                "fc2_experts_weights",
                "fc2_scales",
            ],
            ["moe"],
            name="sparse_qmoe",
            domain="com.microsoft",
            activation_type="swiglu",
            k=TOP_K,
            normalize_routing_weights=0,
            swiglu_fusion=1,
            expert_weight_bits=BITS,
            block_size=BLOCK_SIZE,
            quant_type="int",
        ),
        helper.make_node("Add", ["attn", "moe"], ["hidden"], name="residual_add"),
        helper.make_node("MatMul", ["hidden", "lm_head"], ["logits_no_bias"], name="lm_head"),
        helper.make_node("Add", ["logits_no_bias", "lm_bias"], ["logits"], name="lm_bias"),
    ]

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, ["batch", HEADS, "past", HEAD_DIM]),
        helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, ["batch", HEADS, "past", HEAD_DIM]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", VOCAB]),
        helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, ["batch", HEADS, "total", HEAD_DIM]),
        helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, ["batch", HEADS, "total", HEAD_DIM]),
    ]
    value_info = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        helper.make_tensor_value_info("k", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        helper.make_tensor_value_info("v", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        helper.make_tensor_value_info("attn_mask_bool", TensorProto.BOOL, ["batch", "total"]),
        helper.make_tensor_value_info("attn", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
        helper.make_tensor_value_info("moe", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]),
    ]
    graph = helper.make_graph(nodes, "tiny_deepseek_v2_qmoe_attention", inputs, outputs, initializers, value_info=value_info)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 24), helper.make_operatorsetid("com.microsoft", 1)],
        producer_name="onnx-genai tiny DeepSeek-V2 QMoE fixture",
    )
    model.ir_version = 11

    output.mkdir(parents=True, exist_ok=True)
    for name in ["model.onnx", "model.onnx.textproto", "inference_metadata.yaml", "tokenizer.json", "manifest.json"]:
        (output / name).unlink(missing_ok=True)
    # The committed fixture is git-friendly ONNX protobuf TextFormat
    # (`model.onnx.textproto`), which the runtime loader parses transparently.
    (output / "model.onnx.textproto").write_text(
        text_format.MessageToString(model), encoding="utf-8"
    )
    write_tokenizer(output / "tokenizer.json")
    (output / "inference_metadata.yaml").write_text(
        "model:\n"
        "  attention:\n"
        "    type: multi_head\n"
        f"    num_attention_heads: {HEADS}\n"
        f"    num_kv_heads: {HEADS}\n"
        f"    head_dim: {HEAD_DIM}\n"
        f"  max_sequence_length: {MAX_SEQUENCE}\n"
        "  io:\n"
        "    token_input: input_ids\n"
        "    attention_mask_input: attention_mask\n"
        "    position_ids_input: position_ids\n"
        "    logits_output: logits\n"
        "    kv_inputs:\n"
        "      - past_key_values.0.key\n"
        "      - past_key_values.0.value\n"
        "    kv_outputs:\n"
        "      - present.0.key\n"
        "      - present.0.value\n",
        encoding="utf-8",
    )
    files = {name: (output / name).stat().st_size for name in ["model.onnx.textproto", "inference_metadata.yaml", "tokenizer.json"]}
    manifest = {
        "generator": "tests/fixtures/tiny-deepseek-v2-qmoe-attention/generate.py",
        "seed": SEED,
        "architecture": "DeepSeek-V2-style standard Attention + sparse QMoE",
        "attention_path": ["ai.onnx::RotaryEmbedding", "ai.onnx::Attention"],
        "moe_path": "com.microsoft::QMoE integer int4 sparse top-k",
        "prompt_ids": PROMPT_IDS,
        "expected_tokens": EXPECTED_TOKENS,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
