#!/usr/bin/env python3
"""Convert a binary ``.onnx`` fixture to git-friendly ``.onnx.textproto``.

Weights are inlined into the textproto (no external ``.onnx.data``), because the
runtime loaders convert textproto -> binary bytes in memory with no
model-directory context, so external data could not be resolved.

Usage:
    convert_fixture_to_textproto.py <model.onnx> [<model2.onnx> ...]

For each ``<name>.onnx`` this writes ``<name>.onnx.textproto`` and verifies the
result round-trips back to a valid model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnxscript.ir as ir


def convert(onnx_path: Path) -> Path:
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    model = ir.load(onnx_path)
    # Force external weights (`.onnx.data`) into memory so they are inlined as
    # raw_data in the textproto. The runtime loaders convert textproto -> binary
    # bytes with no model-directory context, so external data must not survive.
    ir.external_data.set_base_dir(model.graph, onnx_path.parent)
    ir.external_data.load_to_model(model)
    out_path = onnx_path.with_suffix(onnx_path.suffix + ".textproto")
    # No external_data arg -> weights are inlined as raw_data in the textproto.
    ir.save(model, out_path, format="textproto")

    # Verify the textproto round-trips to a structurally valid model.
    reloaded = ir.load(out_path, format="textproto")
    n_nodes = sum(1 for _ in reloaded.graph)
    assert n_nodes > 0, f"{out_path} produced an empty graph"
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    failures: list[tuple[str, str]] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        try:
            out = convert(p)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((arg, f"{type(exc).__name__}: {exc}"))
            print(f"SKIP {p}: {type(exc).__name__}: {exc}")
            continue
        print(f"{p} -> {out} ({out.stat().st_size} bytes)")
    if failures:
        print(f"\n{len(failures)} file(s) could not be converted:")
        for name, why in failures:
            print(f"  - {name}: {why}")


if __name__ == "__main__":
    main()
