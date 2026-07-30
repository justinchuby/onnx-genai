#!/usr/bin/env python3
"""One-time migration: positional citations -> content-anchored citations.

For each `path:LINE`, find the DEFINITION enclosing that line and rewrite the
citation as `path::symbol`. Position is consulted exactly once, here, at
migration time. After this runs, nothing in the pipeline reads a line number.

Citations whose enclosing definition cannot be identified are left alone and
reported, because a wrong anchor is worse than an honest positional citation:
a positional citation announces its own staleness by drifting, while a
confidently wrong symbol name looks correct forever.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

POSITIONAL = re.compile(r"`([\w./\-]+\.(?:rs|py|js|css|html|toml|md|sh)):(\d+)(?:-(\d+))?`")

ENCLOSING = [
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union|macro_rules!)\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?([A-Z_][A-Z0-9_]*)\s*:"),
    # `let`/`var` are deliberately absent. A Rust local binding is not a
    # citable definition: anchoring to `let guard = ...` produces a citation
    # that names a variable which any refactor may rename with no review,
    # and the harness would then report a missing symbol in a file that is
    # perfectly correct. Only items that appear in an API surface qualify.
    re.compile(r"^\s*(?:export\s+)?(?:function|class)\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_]\w*)\s*="),
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)"),
    re.compile(r"^#+\s+(.+?)\s*$"),
]


def repo_root() -> Path:
    return Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip())


def tracked(repo: Path) -> set[str]:
    return set(subprocess.run(["git", "ls-files"], cwd=repo,
                              capture_output=True, text=True, check=True).stdout.splitlines())


def resolve(paths: set[str], cited: str) -> str | None:
    if cited in paths:
        return cited
    hits = [t for t in paths if t.endswith("/" + cited)] or \
           [t for t in paths if t.rsplit("/", 1)[-1] == cited]
    return hits[0] if len(hits) >= 1 else None


def enclosing_symbol(lines: list[str], line_no: int) -> tuple[str, str] | None:
    """Resolve the definition a cited line belongs to.

    A citation landing on a doc comment or a blank line belongs to the
    definition BELOW it, not the one above. Walking unconditionally upward
    would anchor every doc-comment citation to the preceding function -- a
    confidently wrong symbol, which is the failure mode this whole harness
    exists to prevent. So: scan down from a comment or blank, up from code.
    """
    idx = min(line_no, len(lines)) - 1
    if idx < 0:
        return None

    def match_at(i: int) -> tuple[str, str] | None:
        for pat in ENCLOSING:
            m = pat.match(lines[i])
            if m:
                if pat.pattern.startswith("^#+"):
                    return (lines[i].strip(), "heading")
                return (m.group(1), "def")
        return None

    stripped = lines[idx].strip()
    is_prose = (not stripped) or stripped.startswith(("//", "/*", "*", "#!"))
    order = range(idx, len(lines)) if is_prose else range(idx, -1, -1)
    for i in order:
        found = match_at(i)
        if found:
            return found
    # A comment block at end of file has nothing below it; fall back upward.
    if is_prose:
        for i in range(idx, -1, -1):
            found = match_at(i)
            if found:
                return found
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: migrate_citations.py <doc.md> [--apply]", file=sys.stderr)
        return 2
    doc = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    repo = repo_root()
    paths = tracked(repo)
    cache: dict[str, list[str]] = {}

    converted: list[str] = []
    skipped: list[str] = []

    def rewrite(m: re.Match) -> str:
        cited, start = m.group(1), int(m.group(2))
        real = resolve(paths, cited)
        if real is None:
            skipped.append(f"{m.group(0)}  file not tracked")
            return m.group(0)
        if real not in cache:
            cache[real] = (repo / real).read_text(errors="replace").splitlines()
        found = enclosing_symbol(cache[real], start)
        if found is None:
            skipped.append(f"{m.group(0)}  no enclosing definition")
            return m.group(0)
        sym, _kind = found
        converted.append(f"{m.group(0)} -> `{real}::{sym}`")
        converted_path = real if "/" in cited or real.count("/") <= 1 else real
        return f"`{converted_path}::{sym}`"

    original = doc.read_text()
    updated = POSITIONAL.sub(rewrite, original)

    print(f"converted {len(converted)} | left positional {len(skipped)}")
    for s in skipped:
        print(f"  SKIP {s}")
    if apply:
        doc.write_text(updated)
        print(f"wrote {doc}")
    else:
        print("(dry run; pass --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
