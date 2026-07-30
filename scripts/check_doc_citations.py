#!/usr/bin/env python3
"""Verify that load-bearing citations in a document point at the code they claim.

WHY THIS EXISTS, AND WHY THE OBVIOUS VERSION IS WORSE THAN NOTHING
------------------------------------------------------------------
An earlier harness checked that every `file.rs:NNN` citation resolved to a
line that EXISTS. It reported `problems 0` on a document in which five
citations pointed at unrelated code: the server's `lib.rs` had grown by ~29
lines and every citation into it had slid down. A citation into a 500-line
file essentially always resolves, so "the line exists" is not a test -- it
is a slow way of confirming the file is non-empty.

Worse, the ambiguity HID the drift. A bare `lib.rs` matches 40 files in this
workspace, so some candidate always had a line 71 and the checker never had
grounds to complain. The property that makes a citation hard for a human to
follow is the same property that makes it impossible for a tool to falsify.

So this checker asserts a SEMANTIC anchor: the cited line must contain an
expected string. Anchors live in the document, immediately beside the claim
they support, as HTML comments -- invisible in rendered Markdown, so there is
no second list to fall out of sync:

    <!-- cite: crates/onnx-genai-engine/src/engine/runtime.rs:1009 = "fn prepare_session_prefix" -->

AND IT REPAIRS RATHER THAN SCOLDS
---------------------------------
Source files move constantly while a document is being written. A checker
that only says "wrong" converts every upstream commit into manual
proofreading, which is how citation rot starts: the check becomes a chore,
the chore gets skipped, the numbers rot anyway.

When the anchor text is found at a different line, this reports the correct
line, and `--fix` rewrites it. Drift stops being a defect and becomes a
mechanical update -- the same move as making a launch command a test rather
than a thing to remember.

Usage:
    python3 scripts/check_doc_citations.py docs/ARCHITECTURE.md [--fix]
Exit status is non-zero if any anchor is unsatisfied and unrepairable.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ANCHOR = re.compile(
    r'<!--\s*cite:\s*(?P<path>[^\s:]+):(?P<line>\d+)\s*=\s*"(?P<text>[^"]+)"\s*-->'
)


def check(doc: pathlib.Path, root: pathlib.Path, fix: bool) -> int:
    text = doc.read_text(encoding="utf-8")
    anchors = list(ANCHOR.finditer(text))

    if not anchors:
        print(f"{doc}: no cite anchors found -- nothing is being verified", file=sys.stderr)
        return 2

    ok = moved = broken = 0
    replacements: list[tuple[str, str]] = []

    for m in anchors:
        rel, want_line, needle = m.group("path"), int(m.group("line")), m.group("text")
        target = root / rel

        if not target.exists():
            print(f"BROKEN   {rel}:{want_line} -- file does not exist")
            broken += 1
            continue

        lines = target.read_text(encoding="utf-8", errors="replace").split("\n")

        if 1 <= want_line <= len(lines) and needle in lines[want_line - 1]:
            ok += 1
            continue

        # The anchor text is the identity of the citation; the number is only
        # its current address. Re-derive the address.
        hits = [i + 1 for i, line in enumerate(lines) if needle in line]

        if len(hits) == 1:
            print(f"MOVED    {rel}:{want_line} -> :{hits[0]}   ({needle!r})")
            moved += 1
            replacements.append((m.group(0), m.group(0).replace(f":{want_line} =", f":{hits[0]} =", 1)))
        elif not hits:
            print(f"BROKEN   {rel}:{want_line} -- {needle!r} no longer appears in the file")
            broken += 1
        else:
            print(f"BROKEN   {rel}:{want_line} -- {needle!r} is ambiguous, at lines {hits}")
            broken += 1

    if fix and replacements:
        for old, new in replacements:
            text = text.replace(old, new, 1)
        doc.write_text(text, encoding="utf-8")
        print(f"\nrewrote {len(replacements)} anchor(s) in {doc}")
        moved = 0

    print(f"\nanchors: {len(anchors)} | ok {ok} | moved {moved} | broken {broken}")
    if moved and not fix:
        print("re-run with --fix to update the moved line numbers")
    return 1 if (moved or broken) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--fix", action="store_true", help="rewrite anchors whose line has moved")
    args = ap.parse_args()
    return check(args.doc, args.root, args.fix)


if __name__ == "__main__":
    sys.exit(main())
