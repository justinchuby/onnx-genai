#!/usr/bin/env python3
"""Generate the deterministic tiny GLM-5.2 fixture(s) used by the native
IndexShare/QMoE regression and the ``--glm-full-attention`` fallback tests.

The generator intentionally imports Mobius's synthetic test configuration
(``tests/_test_configs.py::ALL_CAUSAL_LM_CONFIGS["glm_moe_dsa"]``) so the
fixture follows the exact same exporter path as production GLM-5.2, just at
tiny dimensions. It is pinned to Mobius commit MOBIUS_COMMIT (see below).

Two variants share this script:

* Default (``config.use_dsa=True``): DeepSeek Sparse Attention lowered to two
  ``pkg.nxrt::IndexShare`` nodes per DSA layer, with routed MoE experts fused
  into one ``com.microsoft::QMoE`` node per MoE layer. Native-CUDA/CPU only
  (stock ORT has no ``pkg.nxrt::IndexShare`` implementation).
* ``--full-attention`` (``config.use_dsa=False``, mirroring the Mobius CLI's
  ``--glm-full-attention`` feature): plain dense MLA with no IndexShare nodes
  at all, exported to ``../tiny-glm52-full-attention/`` by default. Still uses
  fused QMoE (a real ``com.microsoft`` contrib op stock ORT does implement),
  so this variant is loadable by both the native engine and stock ORT.

Both variants are written as a self-contained ``model.onnx.textproto`` (weights
inlined as ``raw_data``, no external ``model.onnx.data`` sidecar) using the
same inlining approach as ``scripts/convert_fixture_to_textproto.py``, because
stock ORT's textproto path (``Session::new``) creates the session from an
in-memory byte buffer with no model-directory context and therefore cannot
resolve external data. The binary ``model.onnx`` Mobius writes as an
intermediate artifact is never committed (see ``.gitignore``'s ``*.onnx``
rule); only the ``.textproto`` twin is git-tracked.
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
# Mobius commit that landed the production glm_moe_dsa exporter (DSA/IndexShare
# + --glm-full-attention dense fallback) and the ALL_CAUSAL_LM_CONFIGS tiny
# entry this generator reads. Update when regenerating against a newer commit.
MOBIUS_COMMIT = "d33c33b347bac987cbeff52dca6c1d595ac7780f"


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


def _inline_to_textproto(onnx_path: Path) -> Path:
    """Convert ``onnx_path`` (with external data) to a self-contained,
    git-friendly ``.textproto`` twin with weights inlined as ``raw_data``.

    Mirrors ``scripts/convert_fixture_to_textproto.py`` (which uses
    ``onnxscript.ir``); reimplemented against ``onnx_ir`` here so the whole
    fixture is produced by a single Mobius-native tool invocation.
    """
    model = ir.load(onnx_path)
    ir.external_data.set_base_dir(model.graph, onnx_path.parent)
    ir.external_data.load_to_model(model)
    out_path = onnx_path.with_suffix(onnx_path.suffix + ".textproto")
    ir.save(model, out_path, format="textproto")
    reloaded = ir.load(out_path, format="textproto")
    n_nodes = sum(1 for _ in reloaded.graph)
    assert n_nodes > 0, f"{out_path} produced an empty graph"
    return out_path


def build(*, mobius_root: Path, output_dir: Path, full_attention: bool) -> dict:
    _configure_mobius_imports(mobius_root.resolve())

    from _test_configs import ALL_CAUSAL_LM_CONFIGS, _base_config
    from mobius._builder import build_from_module
    from mobius._configs import QuantizationConfig
    from mobius._registry import registry
    from mobius.integrations.onnx_genai import write_onnx_genai_config
    from mobius.integrations.transformers._config_resolver import _default_task_for_model

    overrides = dict(
        next(overrides for model, overrides, _ in ALL_CAUSAL_LM_CONFIGS if model == "glm_moe_dsa")
    )
    if full_attention:
        overrides["use_dsa"] = False
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT
    # quant_method must be one of the native-QMoE-ABI methods
    # (mobius._weight_utils.supported_qmoe_quantization: "gptq", "awq",
    # "olive") for MoELayer to construct the fused com.microsoft::QMoE node
    # directly -- matching the deepseek_v4_flash_test.py precedent. "gguf"
    # is not in that set and falls back to a portable dense per-expert
    # MatMulNBits representation instead (no QMoE node), which does not
    # match this fixture's advertised emission below.
    #
    # group_size=16 (not 32): MatMulNBits requires block_size <= the linear's
    # in_features (K). This tiny config's smallest quantized Linear is
    # kv_b_proj with in_features=kv_lora_rank=16; a group_size of 32 would
    # exceed it, so the standard MatMulNBits block-quant decomposition pads
    # the (single, undersized) block up to the full block_size before
    # dequantizing, producing a real K=16 vs reconstructed-K=32 shape
    # mismatch when combined with the true (unpadded) activation. Real
    # DeepSeek/GLM-5.2 checkpoints never hit this: their un-shrunk
    # kv_lora_rank (512+) is always far larger than any block_size in use.
    # group_size=16 matches deepseek_v4_flash_test.py's precedent and evenly
    # divides every quantized Linear's in_features in this tiny config
    # (16, 32, 64), so no block is ever padded.
    config.quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="gptq",
        sym=True,
    )

    model_type = "glm_moe_dsa"
    module = registry.get(model_type)(config)
    # build_from_module (not the lower-level registry+task.build combo) is the
    # production path: it also runs optimize_model()'s EP-aware fusion +
    # shape-inference stage (QMoE fusion, GQA/Attention fusion, and populating
    # the logits output's dtype/shape -- required by write_onnx_genai_config's
    # decoder-workflow contract since mobius#554).
    package = build_from_module(
        module,
        config,
        task=_default_task_for_model(model_type),
        execution_provider="cpu",
    )
    rng = np.random.default_rng(SEED)
    for model in package.values():
        _fill_weights(model, rng)

    output = output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "model.onnx",
        "model.onnx.data",
        "model.onnx.textproto",
        "inference_metadata.yaml",
        "tokenizer.json",
        "manifest.json",
    ]:
        (output / name).unlink(missing_ok=True)
    package.save(output, external_data="onnx", check_weights=False)
    write_onnx_genai_config(package, output, config=config)
    _write_tokenizer(output / "tokenizer.json")

    textproto_path = _inline_to_textproto(output / "model.onnx")
    # The binary model.onnx/.data are intermediate artifacts only: .onnx is
    # git-ignored repo-wide and the textproto is now self-contained, so keeping
    # a stale, unreferenced .data sidecar around would be misleading.
    (output / "model.onnx").unlink()
    (output / "model.onnx.data").unlink(missing_ok=True)

    files = {}
    for name in ["model.onnx.textproto", "inference_metadata.yaml", "tokenizer.json"]:
        files[name] = (output / name).stat().st_size
    # Compute emission from the actual built graph rather than assuming it --
    # a hardcoded list here previously went stale silently (see the
    # quant_method note above: an unsupported quant_method quietly produced
    # a dense per-expert MatMulNBits loop with zero QMoE nodes while this
    # list still claimed QMoE was present).
    node_ops = {f"{node.domain}::{node.op_type}" for model in package.values() for node in model.graph}
    # Fixed, human-readable order (not the arbitrary alphabetical order of
    # `node_ops`): IndexShare (DSA-specific) before QMoE (present in both
    # modes), matching this fixture's documented emission order.
    _EMISSION_ORDER = ["pkg.nxrt::IndexShare", "com.microsoft::QMoE"]
    emission = [op for op in _EMISSION_ORDER if op in node_ops]
    expected = _EMISSION_ORDER if not full_attention else ["com.microsoft::QMoE"]
    if emission != expected:
        raise AssertionError(
            f"built graph emission {emission} != expected {expected} "
            f"(full_attention={full_attention}); fixture would not match its "
            "own manifest and test assumptions"
        )
    manifest = {
        "generator": "tests/fixtures/tiny-glm52-qmoe-indexshare/generate.py",
        "mobius_commit": MOBIUS_COMMIT,
        "seed": SEED,
        "architecture": model_type,
        "use_dsa": not full_attention,
        "emission": emission,
        "prompt_ids": [123],
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


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
        default=None,
        help="Defaults to this script's directory (DSA mode) or "
        "../tiny-glm52-full-attention (--full-attention).",
    )
    parser.add_argument(
        "--full-attention",
        action="store_true",
        help="Export config.use_dsa=False (the --glm-full-attention dense MLA "
        "fallback) instead of the default DSA/IndexShare path.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.full_attention:
        output_dir = here.parent / "tiny-glm52-full-attention"
    else:
        output_dir = here

    manifest = build(
        mobius_root=args.mobius_root,
        output_dir=output_dir,
        full_attention=args.full_attention,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
