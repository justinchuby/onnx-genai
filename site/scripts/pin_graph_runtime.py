#!/usr/bin/env python3
"""Replace the pinned Graph plugin's floating CDN URLs with exact versions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPLACEMENTS = {
    "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js": (
        "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
    ),
    "https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.js": (
        "https://cdn.jsdelivr.net/npm/pixi.js@8.19.0/dist/pixi.js"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_dist", type=Path)
    args = parser.parse_args()
    files = sorted(args.graph_dist.rglob("*.js"))
    if not files:
        print(f"Graph build output is missing under {args.graph_dist}", file=sys.stderr)
        return 1

    for floating, exact in REPLACEMENTS.items():
        floating_count = 0
        exact_count = 0
        sources: dict[Path, str] = {}
        for path in files:
            source = path.read_text(encoding="utf-8")
            sources[path] = source
            floating_count += source.count(floating)
            exact_count += source.count(exact)
        if floating_count:
            for path, source in sources.items():
                path.write_text(source.replace(floating, exact), encoding="utf-8")
            exact_count += floating_count
        if exact_count != 2:
            print(
                f"Graph runtime URL {exact!r}: expected 2 occurrences, found {exact_count}",
                file=sys.stderr,
            )
            return 1

    print("Pinned Graph runtime URLs to d3 7.9.0 and pixi.js 8.19.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
