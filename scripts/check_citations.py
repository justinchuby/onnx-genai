#!/usr/bin/env python3
"""Content-anchored citation harness.

A citation in our docs is checked against the CONTENT it names, never against
a position in a file. Inserting unrelated lines above a cited symbol must not
break the citation; renaming or deleting that symbol must.

Three failure classes, deliberately kept distinct in the output, because a
reader who cannot tell "wrong tree" from "never written" loses hours:

    FILE_NOT_IN_GIT   the cited path is not tracked on this branch at all.
                      Either the citation names a document that was never
                      written, or you are standing in the wrong tree.
    SYMBOL_NOT_IN_FILE the file is tracked, but nothing in it defines the
                      named symbol. The symbol was renamed or deleted.
    UNANCHORED        a legacy `file:line` citation. Not an error yet; these
                      are ratcheted down by the manifest and cannot increase.

Anchored citation syntax, inside backticks:

    `crates/onnx-genai-server/src/metrics.rs::prefix_reuse_increments`
    `docs/ARCHITECTURE.md::## 5. Claims`

The harness resolves the symbol to its current line at report time, so a
reader still gets a line number -- it is just never the thing being checked.

Run:
    scripts/check_citations.py docs/ARCHITECTURE.md
    scripts/check_citations.py --self-test      # mutation-proves every class
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

MANIFEST = Path("docs/citations.manifest.json")

# `path/to/file.ext::symbol` -- the anchored form.
ANCHORED = re.compile(r"`([\w./\-]+\.(?:rs|py|js|css|html|toml|md|sh))::([^`]+)`")

# `path/to/file.ext:123` -- the legacy positional form we are ratcheting out.
POSITIONAL = re.compile(r"`([\w./\-]+\.(?:rs|py|js|css|html|toml|md|sh)):(\d+)(?:-\d+)?`")

# A bare mention of a markdown document in prose, e.g. "§4 of perf-baseline.md".
# Catches citations to documents that do not exist, which carry no line number
# and so evade every positional checker.
DOC_MENTION = re.compile(r"(?<![\w./-])([\w-]+\.md)\b")

# Definition sites we recognise. A citation anchors to a DEFINITION, not to any
# textual occurrence -- otherwise a citation could anchor to a comment that
# merely mentions the name, which is the exact laundering we are preventing.
DEFINITION_PATTERNS = [
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+{sym}\b",
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union|type|mod|macro_rules!)\s+{sym}\b",
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?{sym}\s*:",
    r"^\s*{sym}\s*:",                      # struct field / object literal key
    r"^\s*(?:export\s+)?(?:const|let|var|function|class)\s+{sym}\b",
    r"^\s*(?:export\s+)?(?:default\s+)?class\s+{sym}\b",
    r"^\s*def\s+{sym}\b",
    r"^\s*{sym}\s*=",                       # python / js assignment
    r"^\s*\.{sym}\b",                       # css class selector
    r"^#+\s*{sym}\s*$",                     # markdown heading (exact)
]


@dataclass
class Citation:
    doc: Path
    line_no: int
    path: str
    symbol: str | None
    raw: str


@dataclass
class Failure:
    kind: str
    citation: Citation
    detail: str


def tracked_files(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def find_definition(text: str, symbol: str) -> int | None:
    """Return the 1-based line where `symbol` is DEFINED, or None."""
    # A markdown-heading anchor is cited verbatim, e.g. "## 5. Claims".
    if symbol.lstrip().startswith("#"):
        want = symbol.strip()
        for i, line in enumerate(text.splitlines(), start=1):
            if line.strip() == want:
                return i
        return None
    sym = re.escape(symbol)
    compiled = [re.compile(p.format(sym=sym)) for p in DEFINITION_PATTERNS]
    for i, line in enumerate(text.splitlines(), start=1):
        for pat in compiled:
            if pat.search(line):
                return i
    return None


def extract(doc: Path) -> tuple[list[Citation], list[Citation], list[Citation]]:
    """Return (anchored, positional, doc_mentions)."""
    anchored: list[Citation] = []
    positional: list[Citation] = []
    mentions: list[Citation] = []
    seen_mentions: set[tuple[str, int]] = set()
    for n, line in enumerate(doc.read_text().splitlines(), start=1):
        for m in ANCHORED.finditer(line):
            anchored.append(Citation(doc, n, m.group(1), m.group(2), m.group(0)))
        for m in POSITIONAL.finditer(line):
            positional.append(Citation(doc, n, m.group(1), None, m.group(0)))
        for m in DOC_MENTION.finditer(line):
            key = (m.group(1), n)
            if key not in seen_mentions:
                seen_mentions.add(key)
                mentions.append(Citation(doc, n, m.group(1), None, m.group(0)))
    return anchored, positional, mentions


def resolve_path(repo: Path, tracked: set[str], cited: str, symbol: str | None = None) -> str | None:
    """Map a cited path onto a tracked path.

    A bare filename like `loader.rs` is ambiguous -- this repo has several.
    Picking the first candidate is how a citation silently retargets to a
    test file that happens to sort earlier, and the reader is then told the
    symbol is missing from a file the author never meant. So ambiguity is
    resolved BY CONTENT: among the candidates, prefer the ones that actually
    define the cited symbol. If exactly one does, that is the file. This is
    the same principle as the rest of the harness -- names, not positions,
    and never a guess where a check is available.
    """
    if cited in tracked:
        return cited
    candidates = [t for t in tracked if t.endswith("/" + cited)]
    if not candidates:
        candidates = [t for t in tracked if t.rsplit("/", 1)[-1] == cited]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if symbol:
        defining = [
            c for c in candidates
            if find_definition((repo / c).read_text(errors="replace"), symbol) is not None
        ]
        if len(defining) >= 1:
            return defining[0]
    # Ambiguous and undecidable by content: prefer src/ over tests/ so the
    # reported failure names the file the author most plausibly meant.
    non_test = [c for c in candidates if "/tests/" not in c and not c.endswith("tests.rs")]
    return (non_test or candidates)[0]


def check(repo: Path, doc: Path, manifest: dict) -> tuple[list[Failure], dict]:
    tracked = tracked_files(repo)
    anchored, positional, mentions = extract(doc)
    failures: list[Failure] = []

    for c in anchored:
        real = resolve_path(repo, tracked, c.path, c.symbol)
        if real is None:
            failures.append(
                Failure(
                    "FILE_NOT_IN_GIT",
                    c,
                    f"'{c.path}' is not tracked on this branch. The cited "
                    f"symbol may be fine -- the FILE is the problem. Check "
                    f"`git log --all --diff-filter=A -- '*{c.path}'` before "
                    f"assuming it was deleted; it may never have existed.",
                )
            )
            continue
        where = find_definition((repo / real).read_text(errors="replace"), c.symbol)
        if where is None:
            failures.append(
                Failure(
                    "SYMBOL_NOT_IN_FILE",
                    c,
                    f"'{real}' is tracked, but defines no '{c.symbol}'. The "
                    f"file is the right one; the symbol was renamed or removed.",
                )
            )

    for c in mentions:
        if resolve_path(repo, tracked, c.path) is None:
            failures.append(
                Failure(
                    "FILE_NOT_IN_GIT",
                    c,
                    f"prose cites document '{c.path}', which is not tracked on "
                    f"this branch and may never have been written.",
                )
            )

    stats = {
        "anchored": len(anchored),
        "unanchored": len(positional),
        "doc_mentions": len(mentions),
    }

    # Anti-shrink anchors. A harness that enumerates citations from the artefact
    # it checks gets GREENER when citations are deleted. These floors are the
    # only thing standing between us and a document that passes by saying less.
    floor = manifest.get("min_anchored_citations")
    if floor is not None and len(anchored) < floor:
        failures.append(
            Failure(
                "COVERAGE_REGRESSION",
                Citation(doc, 0, str(doc), None, ""),
                f"{len(anchored)} anchored citations, manifest floor is {floor}. "
                f"Citations were deleted rather than repaired. Deleting a "
                f"citation must never be the cheapest way to go green.",
            )
        )
    ceiling = manifest.get("max_unanchored_citations")
    if ceiling is not None and len(positional) > ceiling:
        failures.append(
            Failure(
                "UNANCHORED_RATCHET",
                Citation(doc, 0, str(doc), None, ""),
                f"{len(positional)} positional file:line citations, ratchet "
                f"allows {ceiling}. Positional citations rot silently; convert "
                f"to `path::symbol` rather than raising the ceiling.",
            )
        )
    for required in manifest.get("must_cite_symbols", []):
        if not any(c.symbol == required for c in anchored):
            failures.append(
                Failure(
                    "REQUIRED_CITATION_MISSING",
                    Citation(doc, 0, str(doc), None, ""),
                    f"manifest requires a citation to '{required}'; none found.",
                )
            )
    return failures, stats


def report(failures: list[Failure], stats: dict, doc: Path) -> int:
    print(f"citations in {doc}: anchored {stats['anchored']} | "
          f"positional {stats['unanchored']} | doc-mentions {stats['doc_mentions']}")
    if not failures:
        print("OK - every anchored citation resolves to a definition that exists.")
        return 0
    by_kind: dict[str, list[Failure]] = {}
    for f in failures:
        by_kind.setdefault(f.kind, []).append(f)
    for kind, group in by_kind.items():
        print(f"\n### {kind}: {len(group)}")
        for f in group:
            loc = f"{f.citation.doc}:{f.citation.line_no}" if f.citation.line_no else str(f.citation.doc)
            print(f"  {loc}  {f.citation.raw}")
            print(f"      {f.detail}")
    return 1


# --------------------------------------------------------------------------
# Mutation self-test. Every failure class is proven to FAIL, and the
# content-anchoring property is proven to STAY GREEN under line insertion.
# A green check with no stated mutation is not evidence.
# --------------------------------------------------------------------------

SAMPLE_SRC = """// header
pub fn cited_symbol(x: usize) -> usize {
    x + 1
}
"""
SAMPLE_DOC = "See `src/sample.rs::cited_symbol` for the mapping.\n"


def self_test() -> int:
    results: list[tuple[str, bool, str]] = []

    def run_case(name: str, mutate, expect_kind: str | None):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "src").mkdir()
            (repo / "docs").mkdir()
            (repo / "src" / "sample.rs").write_text(SAMPLE_SRC)
            (repo / "docs" / "d.md").write_text(SAMPLE_DOC)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "src/sample.rs", "docs/d.md"], cwd=repo, check=True)
            mutate(repo)
            failures, _ = check(repo, repo / "docs" / "d.md", {})
            kinds = {f.kind for f in failures}
            if expect_kind is None:
                ok = not failures
                detail = "green as required" if ok else f"unexpectedly red: {kinds}"
            else:
                ok = expect_kind in kinds
                detail = f"raised {kinds}" if ok else f"DID NOT RAISE {expect_kind}; got {kinds or 'nothing'}"
            results.append((name, ok, detail))

    # 0. Control: unmutated tree must be green, or every later case is vacuous.
    run_case("control (unmutated)", lambda r: None, None)

    # 1. THE acceptance criterion for "content-anchored": 20 blank lines above
    #    the cited symbol. A positional checker goes red here; we must not.
    def insert_lines(repo: Path):
        p = repo / "src" / "sample.rs"
        p.write_text("\n" * 20 + p.read_text())
    run_case("insert 20 blank lines above cited symbol", insert_lines, None)

    # 2. Rename the cited symbol.
    def rename(repo: Path):
        p = repo / "src" / "sample.rs"
        p.write_text(p.read_text().replace("cited_symbol", "renamed_symbol"))
    run_case("rename cited symbol", rename, "SYMBOL_NOT_IN_FILE")

    # 3. Delete the cited symbol entirely.
    def delete_symbol(repo: Path):
        (repo / "src" / "sample.rs").write_text("// header\n")
    run_case("delete cited symbol", delete_symbol, "SYMBOL_NOT_IN_FILE")

    # 4. Cite a file that does not exist -- the perf-baseline specimen.
    def missing_file(repo: Path):
        (repo / "docs" / "d.md").write_text("See `docs/never-written.md::intro`.\n")
    run_case("cite file that was never written", missing_file, "FILE_NOT_IN_GIT")

    # 5. The same specimen in PROSE, with no backticks and no line number --
    #    exactly the shape of "§4 of perf-baseline.md".
    def prose_missing(repo: Path):
        (repo / "docs" / "d.md").write_text("As established in section 4 of perf-baseline.md, ...\n")
    run_case("prose mention of nonexistent doc", prose_missing, "FILE_NOT_IN_GIT")

    # 6. Untracked-but-present file: exists on disk, absent from git. This is
    #    how a citation passes locally and fails for everyone who clones.
    def untracked(repo: Path):
        (repo / "docs" / "loose.md").write_text("# loose\n")
        (repo / "docs" / "d.md").write_text("See `docs/loose.md::# loose`.\n")
    run_case("cited file present on disk but untracked", untracked, "FILE_NOT_IN_GIT")

    # 7. Anchoring to a mere MENTION must not satisfy the citation. A comment
    #    naming the symbol is not a definition of it.
    def mention_only(repo: Path):
        (repo / "src" / "sample.rs").write_text("// cited_symbol is described here\n")
    run_case("symbol appears only in a comment", mention_only, "SYMBOL_NOT_IN_FILE")

    # 8. Anti-shrink: deleting the citation must NOT be a way to go green.
    def shrink(repo: Path):
        (repo / "docs" / "d.md").write_text("No citations at all here.\n")
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "docs").mkdir()
        (repo / "docs" / "d.md").write_text("No citations at all here.\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "docs/d.md"], cwd=repo, check=True)
        failures, _ = check(repo, repo / "docs" / "d.md", {"min_anchored_citations": 1})
        ok = any(f.kind == "COVERAGE_REGRESSION" for f in failures)
        results.append((
            "deleting all citations trips the coverage floor",
            ok,
            "raised COVERAGE_REGRESSION" if ok else "WENT GREEN BY SAYING LESS",
        ))

    width = max(len(n) for n, _, _ in results)
    print("MUTATION SELF-TEST")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} mutations behaved as specified.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc", nargs="?", help="markdown document to check")
    ap.add_argument("--self-test", action="store_true", help="mutation-prove the harness")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.doc:
        ap.error("a document is required (or --self-test)")

    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    doc = Path(args.doc)
    if not doc.exists():
        print(f"FILE_NOT_IN_GIT: {doc} does not exist", file=sys.stderr)
        return 1
    mpath = repo / args.manifest
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    failures, stats = check(repo, doc, manifest)
    return report(failures, stats, doc)


if __name__ == "__main__":
    sys.exit(main())
