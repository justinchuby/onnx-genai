#!/usr/bin/env python3
"""Evaluate a Qwen divergence with an independent PyTorch float32 oracle.

The default ``onnx-q4`` mode dequantizes the exact deployed ONNX block-32 int4
weights to float32, installs them into Hugging Face's independent Qwen
implementation, and evaluates the committed teacher-forced prefix.

Example:
  python3 scripts/qwen_q4_f32_oracle.py \
    --case qwen2.5-1.5b \
    --hf-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from transformers import AutoModelForCausalLM


def dequantize_q4(initializers: dict[str, object], base: str) -> torch.Tensor:
    packed = np.asarray(numpy_helper.to_array(initializers[f"{base}.weight_Q4"]))
    scales = np.asarray(
        numpy_helper.to_array(initializers[f"{base}.weight_scales"])
    ).astype(np.float32)
    scales = scales.reshape(packed.shape[0], packed.shape[1])
    quantized = np.empty((*packed.shape[:2], 32), dtype=np.int8)
    quantized[:, :, 0::2] = packed & 15
    quantized[:, :, 1::2] = packed >> 4
    weight = (quantized.astype(np.float32) - 8.0) * scales[:, :, None]
    return torch.from_numpy(weight.reshape(packed.shape[0], -1))


def initializer_array(initializers: dict[str, object], name: str) -> torch.Tensor:
    value = np.asarray(numpy_helper.to_array(initializers[name])).astype(np.float32)
    return torch.from_numpy(value)


def set_parameter(parameter: torch.nn.Parameter, value: torch.Tensor) -> None:
    parameter.data = value.to(dtype=torch.float32)


def install_exact_onnx_weights(model: object, model_path: Path) -> None:
    graph = onnx.load(model_path, load_external_data=True)
    initializers = {value.name: value for value in graph.graph.initializer}

    set_parameter(
        model.model.embed_tokens.weight,
        initializer_array(initializers, "model.embed_tokens.weight"),
    )
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        set_parameter(
            layer.input_layernorm.weight,
            initializer_array(initializers, f"{prefix}.input_layernorm.weight"),
        )
        set_parameter(
            layer.post_attention_layernorm.weight,
            initializer_array(
                initializers, f"{prefix}.post_attention_layernorm.weight"
            ),
        )

        qkv = dequantize_q4(initializers, f"{prefix}.attn.qkv_proj.MatMul")
        q_size = layer.self_attn.q_proj.out_features
        k_size = layer.self_attn.k_proj.out_features
        v_size = layer.self_attn.v_proj.out_features
        set_parameter(layer.self_attn.q_proj.weight, qkv[:q_size])
        set_parameter(layer.self_attn.k_proj.weight, qkv[q_size : q_size + k_size])
        set_parameter(
            layer.self_attn.v_proj.weight,
            qkv[q_size + k_size : q_size + k_size + v_size],
        )

        bias = initializer_array(
            initializers, f"{prefix}.attn.qkv_proj.Add.bias"
        )
        set_parameter(layer.self_attn.q_proj.bias, bias[:q_size])
        set_parameter(
            layer.self_attn.k_proj.bias, bias[q_size : q_size + k_size]
        )
        set_parameter(
            layer.self_attn.v_proj.bias,
            bias[q_size + k_size : q_size + k_size + v_size],
        )
        set_parameter(
            layer.self_attn.o_proj.weight,
            dequantize_q4(initializers, f"{prefix}.attn.o_proj.MatMul"),
        )
        for projection in ("gate", "up", "down"):
            set_parameter(
                getattr(layer.mlp, f"{projection}_proj").weight,
                dequantize_q4(
                    initializers, f"{prefix}.mlp.{projection}_proj.MatMul"
                ),
            )

    final_norm_name = next(
        name
        for name in initializers
        if name.endswith(".final_norm_layernorm.weight")
    )
    set_parameter(
        model.model.norm.weight,
        initializer_array(initializers, final_norm_name),
    )
    set_parameter(
        model.lm_head.weight,
        dequantize_q4(initializers, "lm_head.MatMul"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/parity/native_ort_cuda_golden.json"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("~/.foundry/cache/models").expanduser(),
    )
    parser.add_argument(
        "--weights",
        choices=("onnx-q4", "hf-dense"),
        default="onnx-q4",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text())
    case = next(
        (value for value in golden["models"] if value["key"] == args.case),
        None,
    )
    if case is None:
        parser.error(f"unknown case: {args.case}")
    if "oracle_input_ids" not in case:
        parser.error(f"case {args.case} has no oracle prefix")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if args.weights == "onnx-q4":
        install_exact_onnx_weights(
            model, args.model_root / case["model_path"] / "model.onnx"
        )

    input_ids = torch.tensor([case["oracle_input_ids"]], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    values, indices = torch.topk(logits, 10)
    result = {
        "case": args.case,
        "weights": args.weights,
        "argmax": int(indices[0]),
        "top_logits": [
            [int(token_id), float(value)]
            for token_id, value in zip(indices, values)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
