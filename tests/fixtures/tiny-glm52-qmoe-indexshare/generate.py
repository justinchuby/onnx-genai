#!/usr/bin/env python3
"""Generate the deterministic tiny GLM-5.2 IndexShare + fused-QMoE fixture.

The generator intentionally imports Mobius's synthetic test configuration so
the fixture follows the same exporter path as production GLM-5.2. It is pinned
to Mobius commit 791773b, the current concat/logical IndexShare emission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

SEED = 0
MOBIUS_COMMIT = "791773b"


def _configure_mobius_imports(root: Path) -> None:
    sys.path.insert(0, str(root / "tests"))
    sys.path.insert(0, str(root / "src"))


def _fill_weights(model: ir.Model, rng: np.random.Generator) -> None:
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in initializer.shape)
        if not shape:
            continue
        if initializer.dtype == ir.DataType.FLOAT:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        elif initializer.dtype == ir.DataType.FLOAT16:
            data = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        elif initializer.dtype == ir.DataType.UINT8:
            data = rng.integers(0, 256, size=shape).astype(np.uint8)
        elif initializer.dtype in (ir.DataType.INT64, ir.DataType.INT32):
            dtype = np.int64 if initializer.dtype == ir.DataType.INT64 else np.int32
            data = rng.integers(0, 10, size=shape).astype(dtype)
        else:
            data = rng.standard_normal(shape).astype(np.float32) * 0.02
        initializer.const_value = ir.Tensor(data)


def _write_tokenizer(path: Path) -> None:
    vocab = {str(index): index for index in range(256)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mobius-root",
        type=Path,
        default=Path(os.environ.get("MOBIUS_ROOT", "../mobius")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    _configure_mobius_imports(args.mobius_root.resolve())

    from _test_configs import ALL_CAUSAL_LM_CONFIGS, _base_config
    from mobius._config_resolver import _default_task_for_model
    from mobius._configs import QuantizationConfig
    from mobius._registry import registry
    from mobius.integrations.onnx_genai import write_onnx_genai_config
    from mobius.tasks import get_task

    overrides = dict(
        next(overrides for model, overrides, _ in ALL_CAUSAL_LM_CONFIGS if model == "glm_moe_dsa")
    )
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT
    config.quantization = QuantizationConfig(
        bits=4,
        group_size=32,
        quant_method="gguf",
        sym=True,
    )
    config.fused_quantized_moe = True

    model_type = "glm_moe_dsa"
    module = registry.get(model_type)(config)
    task = get_task(_default_task_for_model(model_type))
    package = task.build(module, config)
    rng = np.random.default_rng(SEED)
    for model in package.values():
        _fill_weights(model, rng)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "model.onnx",
        "model.onnx.data",
        "inference_metadata.yaml",
        "tokenizer.json",
        "manifest.json",
    ]:
        (output / name).unlink(missing_ok=True)
    package.save(output, external_data="onnx", check_weights=False)
    write_onnx_genai_config(package, output, config=config)
    _write_tokenizer(output / "tokenizer.json")

    files = {}
    for name in ["model.onnx", "model.onnx.data", "inference_metadata.yaml", "tokenizer.json"]:
        files[name] = (output / name).stat().st_size
    manifest = {
        "generator": "tests/fixtures/tiny-glm52-qmoe-indexshare/generate.py",
        "mobius_commit": MOBIUS_COMMIT,
        "seed": SEED,
        "architecture": model_type,
        "emission": ["pkg.nxrt::IndexShare", "com.microsoft::QMoE"],
        "prompt_ids": [123],
        "expected_tokens": [62, 164, 59, 205, 48, 166, 27, 9, 221, 190, 123, 108],
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
