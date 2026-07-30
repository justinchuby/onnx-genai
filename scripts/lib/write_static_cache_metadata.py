#!/usr/bin/env python3
"""Write the ``model.io.static_cache`` block into a model's inference metadata.

Mobius exports a static-cache (TensorScatter) model without an ``io:`` block,
but the runtime *requires* one: the scatter control ports ``write_indices`` and
``kv_sequence_length`` are both integer tensors of indistinguishable shape, so
the loader cannot tell them apart from the graph alone and deliberately refuses
to guess (see ``reject_undeclared_static_cache`` in
``crates/onnx-genai-ort/src/decode/io.rs``). Without this block the model fails
to load with:

    graph exposes a TensorScatter static-cache scatter ABI but declares no
    `model.io.static_cache`

This script derives the block by inspecting the actual ONNX graph rather than
assuming a naming convention, then merges it into ``inference_metadata.yaml``.

Usage:
    write_static_cache_metadata.py MODEL_DIR          # write the block
    write_static_cache_metadata.py MODEL_DIR --check  # print it, write nothing

``--check`` is read-only and is how the test suite verifies the derivation
against a known-good model without rebuilding or mutating anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import onnx
except ImportError:  # pragma: no cover - dependency reported by the shell script
    sys.exit("error: the 'onnx' package is required; install Mobius and its dependencies")

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("error: the 'pyyaml' package is required; install Mobius and its dependencies")


# Layer-indexed cache ports, e.g. "key_cache.0" / "updated_key_cache.11".
CACHE_PORT_PATTERNS = {
    "key_cache_inputs": re.compile(r"^key_cache\.(\d+)$"),
    "value_cache_inputs": re.compile(r"^value_cache\.(\d+)$"),
    "key_cache_outputs": re.compile(r"^updated_key_cache\.(\d+)$"),
    "value_cache_outputs": re.compile(r"^updated_value_cache\.(\d+)$"),
}

# The runtime pairs these four lists POSITIONALLY, so every list must be
# ordered by layer index. Sorting the raw strings would order layer 10 before
# layer 2 and silently mis-pair every cache buffer past the ninth layer.
WRITE_INDICES_CANDIDATES = ("write_indices",)
KV_SEQUENCE_LENGTH_CANDIDATES = ("nonpad_kv_seqlen", "kv_sequence_length")


class StaticCacheError(RuntimeError):
    """Raised when the graph is not a usable static-cache export."""


def collect_indexed_ports(names: list[str], pattern: re.Pattern[str]) -> dict[int, str]:
    """Map layer index -> port name for every name matching ``pattern``."""
    found: dict[int, str] = {}
    for name in names:
        match = pattern.match(name)
        if match:
            found[int(match.group(1))] = name
    return found


def pick_first_present(names: list[str], candidates: tuple[str, ...], role: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise StaticCacheError(
        f"graph has no {role} port (looked for: {', '.join(candidates)}). "
        f"Available inputs: {', '.join(sorted(names))}"
    )


def derive_static_cache_spec(model_path: Path) -> dict[str, object]:
    """Derive the ``static_cache`` block from the ONNX graph's port names."""
    # load_external_data=False keeps this fast: the weights are gigabytes and
    # only the graph's input/output names are needed.
    model = onnx.load(str(model_path), load_external_data=False)
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    if not any(
        name in input_names
        for name in WRITE_INDICES_CANDIDATES + KV_SEQUENCE_LENGTH_CANDIDATES
    ):
        raise StaticCacheError(
            f"{model_path} does not expose a static-cache scatter ABI "
            "(no write_indices / nonpad_kv_seqlen input). It was probably built "
            "without --static-cache."
        )

    spec: dict[str, object] = {
        "write_indices_input": pick_first_present(
            input_names, WRITE_INDICES_CANDIDATES, "write_indices"
        ),
        "kv_sequence_length_input": pick_first_present(
            input_names, KV_SEQUENCE_LENGTH_CANDIDATES, "kv_sequence_length"
        ),
    }

    port_sets = {
        role: collect_indexed_ports(
            input_names if role.endswith("_inputs") else output_names, pattern
        )
        for role, pattern in CACHE_PORT_PATTERNS.items()
    }

    layer_indices = sorted(port_sets["key_cache_inputs"])
    if not layer_indices:
        raise StaticCacheError(
            f"{model_path} exposes scatter control ports but no key_cache.N inputs; "
            "the export is incomplete."
        )

    # Every layer must be present in all four lists, or the positional pairing
    # the runtime relies on would be wrong.
    for role, ports in port_sets.items():
        missing = [index for index in layer_indices if index not in ports]
        if missing:
            raise StaticCacheError(
                f"{model_path} is missing {role} for layer(s) "
                f"{', '.join(str(index) for index in missing)}; the export is "
                "incomplete and would mis-pair KV buffers."
            )

    for role in CACHE_PORT_PATTERNS:
        spec[role] = [port_sets[role][index] for index in layer_indices]

    return spec


def derive_io_block(model_path: Path) -> dict[str, object]:
    """Derive the full ``model.io`` block, including token/logits ports."""
    model = onnx.load(str(model_path), load_external_data=False)
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    return {
        "token_input": pick_first_present(input_names, ("input_ids",), "token_input"),
        "logits_output": pick_first_present(output_names, ("logits",), "logits_output"),
        "static_cache": derive_static_cache_spec(model_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write model.io.static_cache into a model's inference_metadata.yaml."
    )
    parser.add_argument("model_dir", type=Path, help="Directory containing model.onnx")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the derived block instead of writing it (read-only).",
    )
    args = parser.parse_args()

    model_path = args.model_dir / "model.onnx"
    metadata_path = args.model_dir / "inference_metadata.yaml"

    if not model_path.is_file():
        return fail(f"no model.onnx in {args.model_dir}")

    try:
        io_block = derive_io_block(model_path)
    except StaticCacheError as error:
        return fail(str(error))

    if args.check:
        yaml.safe_dump({"io": io_block}, sys.stdout, sort_keys=False, default_flow_style=False)
        return 0

    if not metadata_path.is_file():
        return fail(
            f"no inference_metadata.yaml in {args.model_dir}. "
            "Build with '--runtime onnx-genai' (not 'ort-genai'), which is what "
            "emits it."
        )

    with metadata_path.open() as handle:
        metadata = yaml.safe_load(handle) or {}

    model_section = metadata.setdefault("model", {})
    if not isinstance(model_section, dict):
        return fail(f"{metadata_path}: 'model' is not a mapping")

    # Merge rather than replace so a hand-tuned io: block keeps any extra keys,
    # and so re-running the script is idempotent.
    existing_io = model_section.get("io")
    merged_io = dict(existing_io) if isinstance(existing_io, dict) else {}
    merged_io.update(io_block)
    model_section["io"] = merged_io

    with metadata_path.open("w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, default_flow_style=False)

    layer_count = len(io_block["static_cache"]["key_cache_inputs"])  # type: ignore[index]
    print(f"Declared model.io.static_cache for {layer_count} layers in {metadata_path}")
    return 0


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
