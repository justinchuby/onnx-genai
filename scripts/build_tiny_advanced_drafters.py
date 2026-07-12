#!/usr/bin/env python3
"""Build tiny Qwen3.6 MTP and EAGLE-3 ONNX drafter fixtures with Mobius.

Run from the onnx-genai repository root:

    PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
      python scripts/build_tiny_advanced_drafters.py

These random, deterministic models validate graph contracts only. They are not
paired with a target model and cannot produce meaningful speculative tokens.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch

from mobius import build_from_module
from mobius._configs import Eagle3Config, Qwen35MtpConfig
from mobius._testing import make_config
from mobius.models.eagle3 import Eagle3DraftModel
from mobius.models.qwen35_mtp import Qwen35MtpModel
from mobius.tasks import Eagle3DraftTask, Qwen35MtpTask

SEED = 20260712
ROOT = Path(__file__).resolve().parents[1]


def _base_fields() -> dict:
    base = make_config(
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    return {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}


def _mtp_config() -> Qwen35MtpConfig:
    fields = _base_fields()
    fields.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        partial_rotary_factor=0.5,
        vocab_size=32,
        max_position_embeddings=16,
    )
    return Qwen35MtpConfig(**fields)


def _eagle_config() -> Eagle3Config:
    fields = _base_fields()
    fields.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        model_type="llama",
        rope_theta=10_000.0,
        partial_rotary_factor=1.0,
        vocab_size=32,
        draft_vocab_size=16,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    return Eagle3Config(**fields)


def _initializer_tensor(name: str, shape: list[int]) -> torch.Tensor:
    if name.endswith(".bias"):
        return torch.zeros(shape, dtype=torch.float32)
    if "norm" in name and name.endswith(".weight"):
        return torch.ones(shape, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED + sum(ord(ch) for ch in name))
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02


def _materialize_static_reshape_dims(model: ir.Model) -> None:
    """Work around Mobius emitting zero placeholders for known Reshape dims."""
    for node in model.graph:
        if node.op_type != "Reshape" or len(node.inputs) < 2:
            continue
        shape_value = node.inputs[1]
        if shape_value.const_value is None or not node.outputs or node.outputs[0].shape is None:
            continue
        requested = shape_value.const_value.numpy().copy()
        output_shape = node.outputs[0].shape
        if len(requested) != len(output_shape):
            continue
        changed = False
        for index, dimension in enumerate(output_shape):
            if requested[index] == 0 and isinstance(dimension, int):
                requested[index] = dimension
                changed = True
        if changed:
            replacement = ir.Value(
                name=f"{shape_value.name}_{node.name}_materialized",
                shape=shape_value.shape,
                type=shape_value.type,
                const_value=ir.tensor(requested.astype(np.int64)),
            )
            model.graph.initializers[replacement.name] = replacement
            node.replace_input_with(1, replacement)


def _save_fixture(name: str, module, config, task, contract: dict) -> None:
    output_dir = ROOT / "tests" / "fixtures" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.onnx", "model.onnx.data", "manifest.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    package = build_from_module(module, config, task=task, execution_provider="default")
    state = {}
    for param_name, initializer in package["model"].graph.initializers.items():
        shape = [int(dim) for dim in initializer.shape]
        state[param_name] = _initializer_tensor(param_name, shape)
    package.apply_weights(state)
    _materialize_static_reshape_dims(package["model"])
    package.save(str(output_dir), check_weights=True, progress_bar=False)

    model = package["model"]
    manifest = {
        "generator": "scripts/build_tiny_advanced_drafters.py",
        "mobius_root": "/Users/justinc/Documents/GitHub/mobius",
        "seed": SEED,
        "architecture": type(module).__name__,
        "task": type(task).__name__,
        "reshape_workaround": (
            "Materializes known static Reshape dimensions that Mobius da92170 "
            "currently serializes as zero copy-placeholders."
        ),
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
        },
        "inputs": [value.name for value in model.graph.inputs],
        "outputs": [value.name for value in model.graph.outputs],
        "contract": contract,
        "files": {
            filename: (output_dir / filename).stat().st_size
            for filename in ("model.onnx", "model.onnx.data")
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    torch.manual_seed(SEED)

    mtp_config = _mtp_config()
    _save_fixture(
        "tiny-qwen35-mtp",
        Qwen35MtpModel(mtp_config),
        mtp_config,
        Qwen35MtpTask(),
        {
            "inputs_embeds": "[batch, sequence_len, 16]",
            "hidden_states": "[batch, sequence_len, 16]",
            "attention_mask": "[batch, past_sequence_len + sequence_len]",
            "position_ids": "[batch, sequence_len]",
            "past_key_values.0.key": "[batch, 1, past_sequence_len, 8]",
            "past_key_values.0.value": "[batch, 1, past_sequence_len, 8]",
            "mtp_hidden": "[batch, sequence_len, 16]",
            "present.0.key": "[batch, 1, past_sequence_len + sequence_len, 8]",
            "present.0.value": "[batch, 1, past_sequence_len + sequence_len, 8]",
        },
    )

    eagle_config = _eagle_config()
    _save_fixture(
        "tiny-eagle3",
        Eagle3DraftModel(eagle_config),
        eagle_config,
        Eagle3DraftTask(),
        {
            "inputs_embeds": "[batch, sequence_len, 16]",
            "fused_hidden": "[batch, sequence_len, 48]",
            "recycled_hidden": "[batch, sequence_len, 16]",
            "attention_mask": "[batch, past_sequence_len + sequence_len]",
            "position_ids": "[batch, sequence_len]",
            "past_key_values.0.key": "[batch, 1, past_sequence_len, 8]",
            "past_key_values.0.value": "[batch, 1, past_sequence_len, 8]",
            "draft_logits": "[batch, sequence_len, 16]",
            "next_hidden": "[batch, sequence_len, 16]",
            "present.0.key": "[batch, 1, past_sequence_len + sequence_len, 8]",
            "present.0.value": "[batch, 1, past_sequence_len + sequence_len, 8]",
        },
    )


if __name__ == "__main__":
    main()
