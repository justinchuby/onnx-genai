#!/usr/bin/env python3
"""Verify the publish workflow lists every crate after the ones it depends on.

`cargo publish` uploads one crate at a time and resolves each against
crates.io, so a crate listed before something it depends on fails outright --
the required version is not there yet. That is invisible until a dependency
edge changes: the tracer sat mid-list happily for months, then startup tracing
gave the loader a dependency on it and the next release died partway through,
leaving some crates published at the new version and some not.

Dev-dependencies are ignored: they are not resolved when packaging.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/publish.yml"


def published_order() -> list[str]:
    return [
        line.split()[-1]
        for line in WORKFLOW.read_text().splitlines()
        if line.strip().startswith("publish_crate ")
    ]


def versioned_dependencies(crate: str, known: set[str]) -> set[str]:
    manifest = ROOT / "crates" / crate / "Cargo.toml"
    if not manifest.is_file():
        return set()
    # Only the real dependency sections are resolved when packaging.
    body = manifest.read_text().split("[dev-dependencies]")[0]
    found = set()
    for dep in known:
        if dep == crate:
            continue
        match = re.search(rf"^{re.escape(dep)}\s*=\s*(.+)$", body, re.M)
        if match and ("version" in match.group(1) or "workspace = true" in match.group(1)):
            found.add(dep)
    return found


def main() -> int:
    order = published_order()
    if not order:
        print(f"No publish_crate lines found in {WORKFLOW}", file=sys.stderr)
        return 1
    position = {crate: index for index, crate in enumerate(order)}
    problems = []
    for crate in order:
        for dep in versioned_dependencies(crate, set(order)):
            if position[dep] > position[crate]:
                problems.append(
                    f"  {crate} (#{position[crate]}) depends on "
                    f"{dep} (#{position[dep]}), which is published later"
                )
    if problems:
        print(
            "Publish order would fail: a crate is listed before something it "
            "depends on, so cargo cannot resolve it against crates.io.\n"
            + "\n".join(problems)
            + f"\n\nFix by moving the dependency earlier in {WORKFLOW.name}.",
            file=sys.stderr,
        )
        return 1
    print(f"Publish order is consistent for {len(order)} crates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
