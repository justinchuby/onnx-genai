#!/usr/bin/env python3
"""Generate the deterministic tiny DeepSeek-V4 fixture used to prove the
QMoE hash-routed export path (Mobius PR #550) loads and decodes end-to-end
through onnx-genai.

The generator intentionally imports Mobius's synthetic test configuration
(``tests/_test_configs.py::ALL_CAUSAL_LM_CONFIGS["deepseek_v4"]``) so the
fixture follows the exact same exporter path as production DeepSeek-V4, just
at tiny dimensions. It is pinned to Mobius commit MOBIUS_COMMIT (see below).

Scope of this fixture (deliberately the *smallest representative slice*,
matching this repo's DEEPSEEK_CSA_MTP_RUNTIME.md's tracked staged rollout):

* ``compress_ratios`` is left at its default (``None`` -> every layer is
  dense, ratio 0). DeepSeek-V4's compressed/indexer CSA path (ratios 4/128)
  lowers to the same native-only sparse-attention primitives GLM-5.2's DSA
  IndexShare needs, and is tracked separately as its own multi-phase runtime
  effort in ``docs/models/DEEPSEEK_CSA_MTP_RUNTIME.md`` (owner @justinchuby).
  It is explicitly out of scope for this fixture.
* ``num_nextn_predict_layers`` is left at its default (``0``) -> no MTP
  sidecar graph is produced (``build_from_module`` only returns a ``"model"``
  package entry; see ``deepseek_v4_flash_test.py::test_mtp_sidecar_exports_*``
  for the sidecar-producing case, which is also out of scope here).

With both of those left at their (production-default-shape) values, the
built graph uses **only standard ONNX ops plus two real ORT contrib ops**
(``com.microsoft::QMoE`` for the routed experts, ``com.microsoft::MatMulNBits``
for every other quantized Linear) -- no native-only custom op is required, so
this fixture is loadable and runnable by native CPU, native CUDA, *and* stock
ORT alike, unlike the GLM-5.2 DSA/IndexShare fixture.

Both variants of the sibling GLM fixture use the same self-contained
``model.onnx.textproto`` inlining approach; this fixture reuses it verbatim
(see that script's docstring for the stock-ORT external-data rationale).

Regenerating (required two-step recipe -- this script alone is NOT enough):
``write_onnx_genai_config`` reproduces ``model.onnx.textproto``/
``tokenizer.json`` byte-for-byte, but its ``inference_metadata.yaml`` is the
pre-canonicalization document Mobius commit MOBIUS_COMMIT emits -- not the
canonical single-``decoder``-component shape actually committed in this
directory. After running this script, canonicalize with:

    cargo run -p onnx-genai-engine --bin migrate_model_io -- --reemit \\
        <output-dir>

Skipping this reintroduces the ten hand-authored, never-real auxiliary
policy components #1883 removed (each referencing a ``policies/*.onnx``
artifact that was never committed), which fails
``decoder_recognizer_agreement.rs``'s classification matrix and package
loading itself (`crates/onnx-genai-ort/src/loader.rs` eagerly resolves every
declared workflow component's artifact at load time). Verified (2026-08-23):
``generate.py`` + ``migrate_model_io --reemit`` reproduces
``inference_metadata.yaml`` byte-identically to the committed file.
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
# Mobius commit that landed the DeepSeek-V4 dense-MoE->QMoE export (mobius#550)
# and the linear_class routed-expert quantization fix (mobius#562) this
# generator depends on. Update when regenerating against a newer commit.
MOBIUS_COMMIT = "e71f4751791636bc165d67bc09fe03415ac5f416"


def _configure_mobius_imports(root: Path) -> None:
    sys.path.insert(0, str(root / "tests"))
    sys.path.insert(0, str(root / "src"))


def _fill_weights(model: ir.Model, rng: np.random.Generator, *, num_local_experts: int) -> None:
    for name, initializer in model.graph.initializers.items():
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
            if name.endswith(".tid2eid"):
                # DeepSeekV4Gate's hash-routing token-id -> expert-id lookup
                # table (see deepseek_v4.py::DeepSeekV4Gate.__init__ /
                # .forward's `op.Gather(self.tid2eid, input_ids, axis=0)`
                # feeding straight into `op.GatherElements(scores,
                # selected_experts, axis=-1)`). Unlike every other integer
                # initializer here, its values are *not* arbitrary -- they
                # are expert indices and must stay within
                # [0, num_local_experts) or the downstream GatherElements
                # indexes out of range against `scores`' num_local_experts-
                # sized last axis.
                high = num_local_experts
            else:
                high = 10
            data = rng.integers(0, high, size=shape).astype(dtype)
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

    Mirrors ``tests/fixtures/tiny-glm52-qmoe-indexshare/generate.py``'s
    identical helper.
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


def build(*, mobius_root: Path, output_dir: Path) -> dict:
    _configure_mobius_imports(mobius_root.resolve())

    from _test_configs import ALL_CAUSAL_LM_CONFIGS, _base_config
    from mobius._builder import build_from_module
    from mobius._configs import QuantizationConfig
    from mobius._registry import registry
    from mobius.integrations.onnx_genai import write_onnx_genai_config
    from mobius.integrations.transformers._config_resolver import _default_task_for_model

    overrides = dict(
        next(overrides for model, overrides, _ in ALL_CAUSAL_LM_CONFIGS if model == "deepseek_v4")
    )
    config = _base_config(**overrides)
    config.dtype = ir.DataType.FLOAT
    assert config.compress_ratios is None, (
        "this fixture intentionally targets the dense (ratio-0) CSA path; "
        "if ALL_CAUSAL_LM_CONFIGS['deepseek_v4'] ever sets compress_ratios, "
        "this generator needs the same native-op-gap treatment as GLM DSA "
        "before it can claim stock-ORT executability"
    )
    assert config.num_nextn_predict_layers == 0, (
        "this fixture intentionally excludes the MTP sidecar graph; if the "
        "tiny config default ever enables it, build_from_module will return "
        "an extra 'mtp' package entry this generator does not yet handle"
    )
    # gptq/group_size=16 matches deepseek_v4_flash_test.py's
    # test_qmoe_eligible_quantization_fuses_routed_experts_into_one_qmoe_per_layer
    # precedent: the native QMoE ABI's supported quant_method set, and a
    # block_size that evenly divides every quantized Linear's in_features in
    # this tiny config (no padded/undersized block).
    config.quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="gptq",
        sym=True,
    )

    model_type = "deepseek_v4"
    module = registry.get(model_type)(config)
    package = build_from_module(
        module,
        config,
        task=_default_task_for_model(model_type),
        execution_provider="cpu",
    )
    assert set(package) == {"model"}, (
        f"expected a single 'model' package entry (no MTP sidecar), got {sorted(package)}"
    )
    rng = np.random.default_rng(SEED)
    for model in package.values():
        _fill_weights(model, rng, num_local_experts=config.num_local_experts)

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
    (output / "model.onnx").unlink()
    (output / "model.onnx.data").unlink(missing_ok=True)

    files = {}
    for name in ["model.onnx.textproto", "inference_metadata.yaml", "tokenizer.json"]:
        files[name] = (output / name).stat().st_size

    node_ops = {f"{node.domain}::{node.op_type}" for model in package.values() for node in model.graph}
    # No native-only op should ever appear in this dense/no-MTP slice; if one
    # does, the tiny config drifted out of the intentionally-dense/no-MTP
    # scope this fixture claims (see the asserts above) and this generator
    # must be revisited rather than silently shipping an unverified emission
    # claim.
    native_only_ops = {op for op in node_ops if op.startswith("pkg.nxrt::")}
    assert not native_only_ops, (
        f"unexpected native-only ops in a fixture that claims stock-ORT "
        f"executability: {native_only_ops}"
    )
    assert "com.microsoft::QMoE" in node_ops, "expected fused routed-expert QMoE node(s)"
    assert "com.microsoft::MatMulNBits" in node_ops, "expected quantized dense Linear nodes"

    manifest = {
        "generator": "tests/fixtures/tiny-deepseek-v4-qmoe/generate.py",
        "mobius_commit": MOBIUS_COMMIT,
        "seed": SEED,
        "architecture": model_type,
        "compress_ratios": config.compress_ratios,
        "num_nextn_predict_layers": config.num_nextn_predict_layers,
        "emission": sorted(node_ops),
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
        help="Defaults to this script's directory.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir is not None else here

    manifest = build(mobius_root=args.mobius_root, output_dir=output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
