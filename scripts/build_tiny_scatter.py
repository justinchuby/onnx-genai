#!/usr/bin/env python3
"""Build a tiny static-cache (TensorScatter) decoder fixture with Mobius.

Default command from the onnx-genai repo root:

    PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
      python scripts/build_tiny_scatter.py

The model is intentionally tiny and randomly initialized. It uses Mobius'
static-cache path (`CausalLMTask(static_cache=True)`), which emits preallocated
KV cache inputs and `TensorScatter` updates.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from mobius import ArchitectureConfig, build_from_module
from mobius.models.qwen import QwenCausalLMModel
from mobius.tasks import CausalLMTask
import onnx_ir as ir

SEED = 20260712


def _make_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        vocab_size=32,
        max_position_embeddings=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        head_dim=8,
        pad_token_id=0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def _initializer_tensor(name: str, shape: list[int]) -> torch.Tensor:
    if name.endswith(".bias"):
        return torch.zeros(shape, dtype=torch.float32)
    if name.endswith(".weight") and (
        ".norm." in name or name.endswith("norm.weight") or "input_layernorm" in name
    ):
        return torch.ones(shape, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED + sum(ord(ch) for ch in name))
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02


def _apply_deterministic_weights(pkg) -> None:
    model = pkg["model"]
    state = {}
    for name, initializer in model.graph.initializers.items():
        shape = [int(dim) for dim in initializer.shape]
        state[name] = _initializer_tensor(name, shape)
    pkg.apply_weights(state)


def _write_tokenizer(path: Path) -> None:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "the": 6,
        "quick": 7,
        "brown": 8,
        "fox": 9,
        "jumps": 10,
        "over": 11,
        "lazy": 12,
        "dog": 13,
        ".": 14,
        ",": 15,
    }
    for idx in range(16, 32):
        vocab[f"tok{idx}"] = idx

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 2), ("<eos>", 3)],
    )
    tokenizer.save(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/tiny-llm-scatter",
        help="Directory to receive model.onnx, model.onnx.data, tokenizer.json.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16,
        help="Static KV cache length baked into key_cache/value_cache inputs.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["model.onnx", "model.onnx.data", "tokenizer.json", "manifest.json"]:
        path = output_dir / name
        if path.exists():
            path.unlink()

    config = _make_config()
    task = CausalLMTask(static_cache=True, max_seq_len=args.max_seq_len)
    pkg = build_from_module(
        QwenCausalLMModel(config),
        config,
        task=task,
        execution_provider="default",
    )
    _apply_deterministic_weights(pkg)
    pkg.save(str(output_dir), check_weights=True, progress_bar=False)
    # Re-emit as git-friendly textproto with weights inlined, then drop binaries.
    ir.save(pkg["model"], str(output_dir / "model.onnx.textproto"), format="textproto")
    for stale in ("model.onnx", "model.onnx.data"):
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    _write_tokenizer(output_dir / "tokenizer.json")

    manifest = {
        "generator": "scripts/build_tiny_scatter.py",
        "mobius_root": "/Users/justinc/Documents/GitHub/mobius",
        "mobius_static_cache": True,
        "mobius_cli_flags": ["--static-cache", f"--max-seq-len={args.max_seq_len}"],
        "seed": SEED,
        "architecture": "QwenCausalLMModel",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "static_cache_max_seq_len": args.max_seq_len,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "kv_hidden": config.num_key_value_heads * config.head_dim,
        "files": {
            name: (output_dir / name).stat().st_size
            for name in ["model.onnx.textproto", "tokenizer.json"]
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
