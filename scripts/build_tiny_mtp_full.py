#!/usr/bin/env python3
"""Build a tiny target decoder plus matching Qwen3.5 MTP head with Mobius.

Run from the onnx-genai repository root:

    PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
      python scripts/build_tiny_mtp_full.py

The fixture is random and deterministic. It tests MTP orchestration, not model
quality. The target embedding and LM-head matrices are also exported as raw
little-endian f32 files for the runtime MTP proposer.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch

from mobius import ArchitectureConfig, build_from_module
from mobius._configs import Qwen35MtpConfig
from mobius._testing import make_config
from mobius.models.gpt2 import GPT2CausalLMModel
from mobius.models.qwen35_mtp import Qwen35MtpModel
from mobius.tasks import CausalLMTask, Qwen35MtpTask

SEED = 20260712
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "tiny-mtp-full"
HIDDEN = 16
VOCAB = 32


class GPT2CausalLMWithHidden(GPT2CausalLMModel):
    """Expose the post-final-norm hidden tensor consumed by the MTP head."""

    def forward(self, op, input_ids, attention_mask, position_ids, past_key_values=None):
        hidden, present = self.transformer(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return self.lm_head(op, hidden), present, [hidden]


def _target_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        vocab_size=VOCAB,
        max_position_embeddings=16,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="gelu",
        head_dim=8,
        pad_token_id=0,
        rope_type=None,
        output_layer_indices=[0],
    )


def _mtp_config() -> Qwen35MtpConfig:
    base = make_config(
        num_hidden_layers=1,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    fields.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        partial_rotary_factor=0.5,
        vocab_size=VOCAB,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        output_layer_indices=[0],
    )
    return Qwen35MtpConfig(**fields)


def _initializer_tensor(name: str, shape: list[int]) -> torch.Tensor:
    if name.endswith(".bias"):
        return torch.zeros(shape, dtype=torch.float32)
    if "norm" in name and name.endswith(".weight"):
        return torch.ones(shape, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED + sum(ord(ch) for ch in name))
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02


def _materialize_static_reshape_dims(model: ir.Model) -> None:
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


def _build(module, config, task, output_dir: Path, overrides=None):
    package = build_from_module(module, config, task=task, execution_provider="default")
    state = {}
    for name, initializer in package["model"].graph.initializers.items():
        shape = [int(dim) for dim in initializer.shape]
        state[name] = (
            overrides[name].clone()
            if overrides is not None and name in overrides
            else _initializer_tensor(name, shape)
        )
    package.apply_weights(state)
    _materialize_static_reshape_dims(package["model"])
    output_dir.mkdir(parents=True, exist_ok=True)
    package.save(str(output_dir), check_weights=True, progress_bar=False)
    return package["model"], state


def _io(model: ir.Model) -> dict:
    def describe(value) -> dict:
        return {
            "name": value.name,
            "dtype": str(value.dtype),
            "shape": [str(dim) for dim in value.shape],
        }

    return {
        "inputs": [describe(value) for value in model.graph.inputs],
        "outputs": [describe(value) for value in model.graph.outputs],
    }


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    target_config = _target_config()
    mtp_config = _mtp_config()

    target = GPT2CausalLMWithHidden(target_config)
    target_model, target_state = _build(
        target,
        target_config,
        CausalLMTask(),
        OUTPUT,
    )

    embedding = target_state["transformer.wte.weight"].contiguous()
    lm_head = target_state["lm_head.weight"].transpose(0, 1).contiguous()
    (OUTPUT / "embedding.f32").write_bytes(embedding.numpy().astype("<f4").tobytes())
    (OUTPUT / "lm_head.f32").write_bytes(lm_head.numpy().astype("<f4").tobytes())

    mtp = Qwen35MtpModel(mtp_config)
    head_model, _ = _build(
        mtp,
        mtp_config,
        Qwen35MtpTask(),
        OUTPUT / "mtp",
    )

    shutil.copy2(
        ROOT / "tests" / "fixtures" / "tiny-llm" / "tokenizer.json",
        OUTPUT / "tokenizer.json",
    )
    manifest = {
        "generator": "scripts/build_tiny_mtp_full.py",
        "mobius_root": "/Users/justinc/Documents/GitHub/mobius",
        "seed": SEED,
        "purpose": "Random deterministic target+MTP fixture for end-to-end orchestration tests.",
        "config": {
            "target_architecture": "GPT2CausalLMWithHidden",
            "mtp_architecture": "Qwen35MtpModel",
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "target_num_key_value_heads": 2,
            "mtp_num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["full_attention"],
            "output_layer_indices": [0],
        },
        "target": _io(target_model),
        "mtp_head": _io(head_model),
        "runtime": {
            "target_hidden_output": "hidden_states.0",
            "embedding_weights": "embedding.f32",
            "embedding_layout": "[vocab, hidden]",
            "lm_head_weights": "lm_head.f32",
            "lm_head_layout": "[hidden, vocab]",
            "mtp_head_model": "mtp/model.onnx",
            "mtp_kv_mode": "HiddenThreaded",
        },
    }
    manifest["files"] = {
        str(path.relative_to(OUTPUT)): path.stat().st_size
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file()
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
