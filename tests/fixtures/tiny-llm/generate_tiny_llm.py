#!/usr/bin/env python3
"""Generate the tiny decoder-only ONNX fixture using Mobius.

Default command from the onnx-genai repo root:

    PYTHONPATH=/Users/justinc/Documents/GitHub/mobius/src \
      python tests/fixtures/tiny-llm/generate_tiny_llm.py

The model is intentionally tiny and randomly initialized: vocab=32,
hidden=16, one GPT-2-style decoder layer, two attention heads, max sequence
length 16. It is only for smoke tests, not quality evaluation.
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
from mobius.models.gpt2 import GPT2CausalLMModel

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
        hidden_act="gelu",
        head_dim=8,
        pad_token_id=0,
        rope_type=None,
    )


def _initializer_tensor(name: str, shape: list[int]) -> torch.Tensor:
    if name.endswith(".bias"):
        return torch.zeros(shape, dtype=torch.float32)
    if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
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
        default=Path(__file__).resolve().parent,
        help="Directory to receive model.onnx, model.onnx.data, tokenizer.json.",
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
    pkg = build_from_module(GPT2CausalLMModel(config), config, execution_provider="default")
    _apply_deterministic_weights(pkg)
    pkg.save(str(output_dir), check_weights=True, progress_bar=False)
    _write_tokenizer(output_dir / "tokenizer.json")

    manifest = {
        "generator": "tests/fixtures/tiny-llm/generate_tiny_llm.py",
        "mobius_root": "/Users/justinc/Documents/GitHub/mobius",
        "seed": SEED,
        "architecture": "GPT2CausalLMModel",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "files": {
            name: (output_dir / name).stat().st_size
            for name in ["model.onnx", "model.onnx.data", "tokenizer.json"]
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
