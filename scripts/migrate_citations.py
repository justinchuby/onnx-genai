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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_context  # noqa: E402

# Was a local mirror of tree_context.CANNOT_RUN alongside a private repo_root().
# The duplication was disclosed and deferred as scope creep, which was the right
# call for that commit and became the wrong state to leave behind: MEASURED IN A
# REAL ARCHIVE EXTRACT, this file exited 1 WITH A RAW TRACEBACK while its three
# siblings exited 2. The crew has been told, and is repeating to reviewers, that
# "four python instruments exit 2". Three did.
#
# That is the exact failure tree_context.repo_root's own docstring forbids -- "a
# crash and a finding must never print the same thing" -- so the sentence was
# written in this directory and contradicted two files away. A reviewer in an
# extract would have read a traceback as a defect in the branch.
CANNOT_RUN = tree_context.CANNOT_RUN

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
    if not doc.exists():
        print(f"CANNOT RUN: {doc} does not exist.", file=sys.stderr)
        print("  This is NOT a finding about the document -- there is no "
              "document. Exit 1 here would be indistinguishable from a real "
              "conversion failure.", file=sys.stderr)
        return CANNOT_RUN
    # tree_context.repo_root() derives the tree from THIS SCRIPT'S location and
    # never from the caller's cwd. That is a real behaviour change from the
    # deleted private copy, which used cwd, and it is the intended one: it is
    # what all four siblings already do, so the four instruments now agree on
    # which tree they are talking about. SCOPED CLAIM, AND THE SCOPE IS LOAD-
    # BEARING: they agree on TREE RESOLUTION only. They do not share a document
    # interface at all -- check_provenance.py and check_provenance_wire.py have
    # fixed corpora and ignore a path argument entirely. Measuring all four with
    # one probe and reading their different answers as different SAFETY is a
    # mistake this file's author has already made once. The property is a known sharp edge and
    # is documented at tree_context.repo_root; it is not rediscovered here.
    try:
        repo = tree_context.repo_root()
    except tree_context.NoWorktree as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        print("  This is NOT a finding about the document. The tool could not "
              "locate a git worktree, so it never read anything to have an "
              "opinion about.", file=sys.stderr)
        return CANNOT_RUN
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
    # Computed for the side effect of populating `converted` and `skipped`,
    # which is what the dry-run report prints. The result is intentionally
    # never written -- see the refusal below.
    POSITIONAL.sub(rewrite, original)

    print(f"converted {len(converted)} | left positional {len(skipped)}")
    for s in skipped:
        print(f"  SKIP {s}")
    if apply:
        # --apply IS DISABLED. This is a refusal, not a bug.
        #
        # This script rewrites `path:NNN` into `path::symbol` INSIDE NORMATIVE
        # DOCUMENTS. It has no quote-awareness of any kind -- no fence
        # handling, no blockquote handling, nothing. Measured, not assumed:
        # grep -ciE 'fence|```|backtick|blockquote' over this file returns 0.
        #
        # That is fatal for a WRITER, and the distinction matters. A
        # frame-blind READER produces a false alarm and costs someone an
        # hour. A frame-blind WRITER produces a fabricated fact and ships it.
        # The live specimen (found by c0de4c2e, generalised by 086345a5):
        #
        #   IMPLEMENTATION-REVIEW.md:142 contains the prose
        #     "README.md cites driver.rs:956, but that file has only 912 lines."
        #
        # That sentence is an OBITUARY for a defect that is already dead.
        # Point this script at it and `driver.rs:956` matches POSITIONAL,
        # resolves against today's driver.rs, and gets rewritten into a
        # confident, present-tense, symbol-anchored citation THAT NOBODY
        # WROTE -- destroying the historical record and manufacturing a claim
        # in its place. The document would then assert, in our own citation
        # format, the opposite of what its author meant.
        #
        # Two further defects, both mine, both measured, either one enough:
        #   - it ENUMERATES from the index (git ls-files, :46) and READS from
        #     the working tree (:116, :126). Those are two different trees.
        #     It can rewrite a citation using bytes that exist in no commit.
        #   - 141 lines, ZERO tests. It has already written one
        #     past-end-of-file citation into README.md.
        #
        # It is referenced by NOTHING at HEAD -- no CI, no doc, no script
        # (control: check_citations.py is referenced by 3). So this refusal
        # cannot break a caller, because there is no caller. An unreferenced
        # writer with no tests and no quote-awareness is not a tool, it is a
        # loaded gun on a shelf, and the safety belongs ON THE GUN rather
        # than in a sentence in a document asking people not to touch it.
        #
        # TO RE-ENABLE, the bar is a test, not a judgement call: a case
        # proving a citation inside a fence and a citation inside a
        # blockquote are both LEFT ALONE, plus the obituary above as a
        # regression fixture.
        print(
            "REFUSING TO WRITE. --apply is disabled by construction.\n"
            "  reason: this rewriter has no fence- or blockquote-awareness, so it\n"
            "          cannot distinguish a CITATION from PROSE QUOTING a citation,\n"
            "          and it would rewrite a quoted dead defect into a live claim.\n"
            "  reason: it enumerates from the index but reads the working tree.\n"
            "  reason: 141 lines, zero tests, and it writes to normative documents.\n"
            "  The dry run above is still trustworthy -- it only READS.\n"
            "  Apply its suggestions by hand, or fix the quote-awareness first.",
            file=sys.stderr,
        )
        return CANNOT_RUN
    print("(dry run only; --apply is disabled -- see the refusal notice in main())")
    return 0


if __name__ == "__main__":
    sys.exit(main())
