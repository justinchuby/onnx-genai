#!/usr/bin/env python3
"""Generate an ``inference_metadata.yaml`` from a model's ``genai_config.json``.

Background (#384)
-----------------
The native decode path (and the seq-major KV floor measurements in
``mobius_seqmajor_growth_parity_native_cuda`` and its qwen14b sibling) requires
each model directory to carry an ``inference_metadata.yaml`` that declares the
graph I/O contract explicitly: which port drives autoregressive execution, the
attention-mask input, the logits output, and the positionally paired
past/present KV port lists. Exports produced by onnxruntime-genai ship a
``genai_config.json`` but **not** this metadata file, so every such model was
non-native-loadable until the file was hand-written -- the #384 gap.

This script closes that gap reproducibly: it reads the authoritative
``genai_config.json`` and emits the metadata deterministically, so a one-off
local fix (e.g. the qwen14b-zp export used for the KV floor measurement) becomes
a repeatable step for any onnxruntime-genai export.

Sources (all read from ``genai_config.json``)
---------------------------------------------
* ``model.context_length``               -> ``model.max_sequence_length``
* ``model.decoder.inputs.input_ids``     -> ``io.token_input``
* ``model.decoder.inputs.attention_mask``-> ``io.attention_mask_input``
* ``model.decoder.outputs.logits``       -> ``io.logits_output``
* ``model.decoder.inputs.past_key_names`` / ``past_value_names`` (``%d`` templates)
  expanded over ``model.decoder.num_hidden_layers`` -> ``io.kv_inputs``
* ``model.decoder.outputs.present_key_names`` / ``present_value_names``
  expanded the same way                  -> ``io.kv_outputs``

KV ports are emitted key-then-value, interleaved per layer
(``past_key_values.0.key``, ``past_key_values.0.value``, ``...1.key``, ...),
matching the order the native decode binding layer pairs them.

Usage
-----
    python scripts/gen_inference_metadata.py MODEL_DIR [--output PATH] [--force]
    python scripts/gen_inference_metadata.py MODEL_DIR --stdout

By default writes ``MODEL_DIR/inference_metadata.yaml`` and refuses to overwrite
an existing file unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise SystemExit(f"genai_config.json: missing required key '{key}' in {where}")
    return mapping[key]


def _expand(template: str, num_layers: int, template_name: str) -> list[str]:
    """Expand a ``%d`` name template over ``[0, num_layers)``."""
    if "%d" not in template:
        raise SystemExit(
            f"genai_config.json: {template_name} '{template}' has no '%d' layer "
            "placeholder; cannot expand per-layer KV port names"
        )
    return [template.replace("%d", str(layer)) for layer in range(num_layers)]


def _interleave(keys: list[str], values: list[str]) -> list[str]:
    """Key-then-value per layer: k0, v0, k1, v1, ... matching binding pairing."""
    ports: list[str] = []
    for key, value in zip(keys, values):
        ports.append(key)
        ports.append(value)
    return ports


def build_metadata(config: dict) -> dict:
    model = _require(config, "model", "root")
    decoder = _require(model, "decoder", "model")
    inputs = _require(decoder, "inputs", "model.decoder")
    outputs = _require(decoder, "outputs", "model.decoder")
    num_layers = int(_require(decoder, "num_hidden_layers", "model.decoder"))
    if num_layers < 1:
        raise SystemExit(f"genai_config.json: num_hidden_layers must be >= 1, got {num_layers}")

    context_length = int(_require(model, "context_length", "model"))

    token_input = _require(inputs, "input_ids", "model.decoder.inputs")
    attention_mask_input = _require(inputs, "attention_mask", "model.decoder.inputs")
    logits_output = _require(outputs, "logits", "model.decoder.outputs")

    past_key_tmpl = _require(inputs, "past_key_names", "model.decoder.inputs")
    past_value_tmpl = _require(inputs, "past_value_names", "model.decoder.inputs")
    present_key_tmpl = _require(outputs, "present_key_names", "model.decoder.outputs")
    present_value_tmpl = _require(outputs, "present_value_names", "model.decoder.outputs")

    kv_inputs = _interleave(
        _expand(past_key_tmpl, num_layers, "past_key_names"),
        _expand(past_value_tmpl, num_layers, "past_value_names"),
    )
    kv_outputs = _interleave(
        _expand(present_key_tmpl, num_layers, "present_key_names"),
        _expand(present_value_tmpl, num_layers, "present_value_names"),
    )

    return {
        "max_sequence_length": context_length,
        "token_input": token_input,
        "attention_mask_input": attention_mask_input,
        "logits_output": logits_output,
        "kv_inputs": kv_inputs,
        "kv_outputs": kv_outputs,
    }


def render_yaml(meta: dict) -> str:
    """Render the metadata with the exact 2-space / 6-space list layout the
    hand-written reference files use, without a YAML dependency."""
    lines: list[str] = []
    lines.append("# Generated by scripts/gen_inference_metadata.py from genai_config.json.")
    lines.append("# Regenerate with: python scripts/gen_inference_metadata.py MODEL_DIR --force")
    lines.append("model:")
    lines.append(f"  max_sequence_length: {meta['max_sequence_length']}")
    lines.append("  io:")
    lines.append(f"    token_input: {meta['token_input']}")
    lines.append(f"    attention_mask_input: {meta['attention_mask_input']}")
    lines.append(f"    logits_output: {meta['logits_output']}")
    lines.append("    kv_inputs:")
    for name in meta["kv_inputs"]:
        lines.append(f"      - {name}")
    lines.append("    kv_outputs:")
    for name in meta["kv_outputs"]:
        lines.append(f"      - {name}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="Model directory containing genai_config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: MODEL_DIR/inference_metadata.yaml)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing a file")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args(argv)

    config_path = args.model_dir / "genai_config.json"
    if not config_path.is_file():
        raise SystemExit(f"no genai_config.json in {args.model_dir}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    meta = build_metadata(config)
    text = render_yaml(meta)

    if args.stdout:
        sys.stdout.write(text)
        return 0

    output = args.output or (args.model_dir / "inference_metadata.yaml")
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing {output} (pass --force)")
    output.write_text(text, encoding="utf-8")
    num_layers = len(meta["kv_inputs"]) // 2
    print(
        f"wrote {output} (max_sequence_length={meta['max_sequence_length']}, "
        f"{num_layers} layers, {len(meta['kv_inputs'])} kv ports)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
