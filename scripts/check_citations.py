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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_context  # noqa: E402

MANIFEST = Path("docs/citations.manifest.json")

# `:123` -- the CONTINUATION form: a bare line reference that inherits its file
# from an earlier citation. This is a deictic: its meaning depends entirely on
# what was written before it, so it breaks when the surrounding prose is
# reordered, and neither ANCHORED nor POSITIONAL matches it. Until this pattern
# existed the checker could not see these citations AT ALL -- they were not
# counted, not ratcheted, and not resolved, so a green run reporting
# "positional 0" was measuring a strict subset of the document's citations
# while reading as if it had covered all of them.
CONTINUATION = re.compile(r"`:(\d+)(?:-\d+)?`")

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

# Inverted forms of the STRONG definition patterns above, used to build a
# repo-wide index of "which files define a symbol of this name".
#
# WHY ONLY THE STRONG FORMS: the loose patterns above (`{sym}:` and `{sym} =`)
# are correct for CONFIRMING a named symbol but useless for DISCOVERING names --
# inverted, they match every struct field, every object key and every
# assignment in the repository, and an index full of noise cannot distinguish
# a genuine ambiguity from a coincidence. Ambiguity is only interesting for
# named entities that a sentence can be *about*.
DEFINITION_CAPTURE = [
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+(\w+)\b"),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union|type|mod|macro_rules!)\s+(\w+)\b"),
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?(\w+)\s*:"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var|function|class)\s+(\w+)\b"),
    re.compile(r"^\s*def\s+(\w+)\b"),
]

INDEXED_SUFFIXES = (".rs", ".py", ".js", ".css", ".html", ".toml", ".sh")

# Above this many definitions, a name is a LANGUAGE CONVENTION rather than a
# confusable: `main` is defined in 92 files here, `load` in 22. Nobody reads
# `main.rs::main` and wonders which `main` is meant, so reporting those buries
# the two- and three-way collisions that are genuinely mistakable -- and a
# checker that cries wolf gets its assertions loosened, which is how a
# safeguard dies. The danger is a SMALL number of same-named owners, which is
# precisely the shape that produced this check: `resource_snapshot` on
# EngineDriver and on Engine, in two crates, both real, one meant.
AMBIGUITY_MAX_OWNERS = 3


def definition_index(repo: Path, tracked: set[str]) -> dict[str, set[str]]:
    """symbol -> {files that define it}. One pass over tracked sources.

    This exists to answer a question the per-citation check structurally
    cannot. `driver.rs::resource_snapshot` resolves perfectly AND a different
    `resource_snapshot` exists in another crate. The citation is correct; a
    sentence naming the wrong owning type beside it is not -- and nothing in a
    resolve-then-confirm check can see that, because both halves pass.

    Reporting the collision is the most a citation harness can do here. It
    cannot know which one the prose meant. It can only refuse to let the author
    believe the question was asked and answered.
    """
    index: dict[str, set[str]] = {}
    for rel in tracked:
        if not rel.endswith(INDEXED_SUFFIXES):
            continue
        try:
            text = (repo / rel).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            for pat in DEFINITION_CAPTURE:
                m = pat.match(line)
                if m:
                    index.setdefault(m.group(1), set()).add(rel)
                    break
    return index


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


def extract(doc: Path) -> tuple[list[Citation], list[Citation], list[Citation], list[Citation]]:
    """Return (anchored, positional, doc_mentions, continuations)."""
    anchored: list[Citation] = []
    positional: list[Citation] = []
    mentions: list[Citation] = []
    continuations: list[Citation] = []
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
        # A continuation inherits its file from the nearest citation to its LEFT
        # ON THE SAME LINE. That rule is deliberately local and conservative: a
        # continuation whose antecedent is in another paragraph is REFUSED
        # rather than attributed by proximity, because guessing an antecedent
        # is how a checker starts fabricating the evidence it was built to
        # audit. `path` carries the inherited file, or None when unattributable.
        for m in CONTINUATION.finditer(line):
            antecedents = [
                (a.start(), a.group(1))
                for a in list(ANCHORED.finditer(line)) + list(POSITIONAL.finditer(line))
                if a.start() < m.start()
            ]
            inherited = max(antecedents)[1] if antecedents else None
            continuations.append(Citation(doc, n, inherited, None, m.group(0)))
    return anchored, positional, mentions, continuations


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
            if find_definition(source_text(repo, c), symbol) is not None
        ]
        if len(defining) == 1:
            return defining[0]
        if len(defining) > 1:
            # Several candidates define it. Returning one would be a GUESS
            # wearing the costume of a content check -- the caller cannot tell
            # a decided answer from an arbitrary one. Prefer the non-test file
            # so the AMBIGUOUS_SYMBOL report names something useful, and let
            # that report -- not this silent pick -- carry the uncertainty.
            non_test_defining = [
                d for d in defining if "/tests/" not in d and not d.endswith("tests.rs")
            ]
            return (non_test_defining or defining)[0]
    # Ambiguous and undecidable by content: prefer src/ over tests/ so the
    # reported failure names the file the author most plausibly meant.
    non_test = [c for c in candidates if "/tests/" not in c and not c.endswith("tests.rs")]
    return (non_test or candidates)[0]


def check(repo: Path, doc: Path, manifest: dict) -> tuple[list[Failure], dict]:
    # Cleared per run. The self-test invokes check() repeatedly against
    # different throwaway repositories, and a set that survived between them
    # would carry one repo's paths into another's divergence report -- a
    # checker leaking state into its own evidence.
    CITED_SOURCES_READ.clear()
    tracked = tracked_files(repo)
    index = definition_index(repo, tracked)
    anchored, positional, mentions, continuations = extract(doc)
    failures: list[Failure] = []
    # Non-fatal observations. A deictic citation is a real defect but a slow
    # one, and turning 32 of them red on demo night would buy a developer
    # nothing while blocking everyone. They are printed on every run and frozen
    # by a ratchet so the count can only fall -- the same shape used for line
    # drift: assert identity fatally, report drift non-fatally.
    reports: list[Failure] = []

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
        where = find_definition(source_text(repo, real), c.symbol)
        if where is None:
            failures.append(
                Failure(
                    "SYMBOL_NOT_IN_FILE",
                    c,
                    f"'{real}' is tracked, but defines no '{c.symbol}'. The "
                    f"file is the right one; the symbol was renamed or removed.",
                )
            )
            continue
        owners = index.get(c.symbol, set())
        if 2 <= len(owners) <= AMBIGUITY_MAX_OWNERS:
            others = sorted(o for o in owners if o != real)
            reports.append(
                Failure(
                    "AMBIGUOUS_SYMBOL",
                    c,
                    f"'{c.symbol}' is also defined in {', '.join(others)}. This "
                    f"citation resolves correctly to '{real}' -- that is exactly "
                    f"the danger, because the prose beside it can name the wrong "
                    f"owner and every check here still passes.",
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

    # Positional citations were previously COUNTED and ratcheted but never
    # resolved -- not even for path existence. That made the ratchet a tally of
    # unchecked text, not a shrinking pile of checked-but-legacy citations. All
    # five in ARCHITECTURE.md turned out to name crate-relative fragments
    # (`cli/src/lib.rs`) that match no tracked path at all, and one basename
    # (`governor.rs`) that matches two different crates. Resolve them too: a
    # citation the tool cannot follow must never be counted as one it checked.
    for c in positional:
        if resolve_path(repo, tracked, c.path) is None:
            failures.append(
                Failure(
                    "UNANCHORED_UNRESOLVABLE",
                    c,
                    f"positional citation '{c.raw}' names a path that is not "
                    f"tracked on this branch. A line number cannot rescue a "
                    f"path that does not exist; re-anchor as `path::symbol`.",
                )
            )

    # A continuation citation is checked as far as it CAN be checked and no
    # further. Where the antecedent is known we assert the line is inside the
    # file; where it is not, we refuse to attribute it and say so. Neither is a
    # content check -- a line number that fits is not a line number that is
    # right -- so these are reported honestly as line-anchored, never counted
    # among the resolved citations.
    for c in continuations:
        if c.path is None:
            reports.append(
                Failure(
                    "CONTINUATION_UNATTRIBUTABLE",
                    c,
                    f"continuation citation '{c.raw}' has no citation to its "
                    f"left on the same line, so the file it refers to is "
                    f"decided by surrounding prose. Reordering the paragraph "
                    f"silently repoints it. Write the path and symbol in full.",
                )
            )
            continue
        real = resolve_path(repo, tracked, c.path)
        if real is None:
            continue  # the antecedent itself is already reported above
        n_lines = len(source_text(repo, real).splitlines())
        cited_line = int(CONTINUATION.match(c.raw).group(1))
        if cited_line > n_lines:
            failures.append(
                Failure(
                    "CONTINUATION_OUT_OF_RANGE",
                    c,
                    f"continuation citation '{c.raw}' inherits '{real}', which "
                    f"has only {n_lines} lines. The line it names does not exist.",
                )
            )

    stats = {
        "anchored": len(anchored),
        "unanchored": len(positional),
        "doc_mentions": len(mentions),
        "continuations": len(continuations),
        "reports": reports,
    }

    # Reconcile the bytes this verdict rests on against what we ship. The
    # document is included alongside the sources: a citation check is a claim
    # about a RELATION between two files, and either one being a draft makes
    # the relation a draft. Scoped to the inputs actually read -- a global
    # "N files uncommitted" banner on a fourteen-agent branch is never zero and
    # has stopped carrying information.
    doc_rel = doc
    try:
        doc_rel = doc.resolve().relative_to(repo.resolve())
    except ValueError:
        pass
    stats["divergence"] = tree_context.divergence_report(
        repo, sorted(CITED_SOURCES_READ | {str(doc_rel)})
    )

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
    cont_ceiling = manifest.get("max_continuation_citations")
    if cont_ceiling is not None and len(continuations) > cont_ceiling:
        failures.append(
            Failure(
                "CONTINUATION_RATCHET",
                Citation(doc, 0, str(doc), None, ""),
                f"{len(continuations)} line-anchored continuation citations, "
                f"ratchet allows {cont_ceiling}. These inherit their file from "
                f"neighbouring prose and rot when it is reordered; write the "
                f"path and symbol in full rather than raising the ceiling.",
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


# Paths whose BYTES were used to justify a citation verdict. Deliberately not
# every file this script opens: `definition_index` reads the whole tracked tree,
# but its output can only ADD ambiguity reports and is never consulted to
# suppress a failure, so divergence there can make the run noisier and never
# quieter. These paths are different -- a citation lives or dies on them.
CITED_SOURCES_READ: set[str] = set()


def source_text(repo: Path, rel: str) -> str:
    """Read a cited source file, RECORDING that a verdict now depends on it.

    This still reads the working tree, and that is a deliberate choice rather
    than an oversight. Reading HEAD instead would go red on the entirely normal
    commit that adds a symbol and cites it in the same change, and a guard that
    fails on correct work is a guard that gets switched off within a day.

    What was actually wrong was never WHICH tree got read -- it was that the
    result never said. So the read is unchanged and the DISCLOSURE is the fix:
    every path that backed a verdict is remembered here and reconciled against
    HEAD before the verdict prints.
    """
    CITED_SOURCES_READ.add(rel)
    return (repo / rel).read_text(errors="replace")


def container_census(doc: Path) -> dict[str, int]:
    """Count positional citations by the container they sit in.

    MEASURED, not assumed -- I published the opposite of this and was wrong.
    I told the crew this harness "strips fenced blocks". It does not. There
    is no fence handling anywhere in this file. The real mechanism is that
    POSITIONAL requires INLINE-CODE BACKTICKS, and most fenced content (ASCII
    diagrams, mermaid labels, pasted terminal output) carries none. So:

      fenced, no backticks   -> invisible  (not protected -- just unmatched)
      fenced, WITH backticks -> COUNTED    (the fence protects NOTHING)
      blockquoted            -> COUNTED
      plain prose            -> COUNTED

    That difference matters because it inverts the remedy. If a fence were an
    exemption, wrapping a quoted finding in one would be a legal fix. It is
    not. A fence is not a shield here and nobody should be told it is.
    """
    census = {"blockquoted": 0, "fenced": 0, "prose": 0}
    in_fence = False
    for line in doc.read_text().splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        hits = len(POSITIONAL.findall(line))
        if not hits:
            continue
        if in_fence:
            census["fenced"] += hits
        elif line.lstrip().startswith(">"):
            census["blockquoted"] += hits
        else:
            census["prose"] += hits
    return census


def tree_qualifier() -> str:
    """The tree this verdict is about, formatted to sit ON the verdict line.

    The banner already prints the toplevel -- at the TOP, dozens of lines
    above the verdict. That is not good enough, and the reason is measured:
    of the 179 anchored citations in docs/ARCHITECTURE.md, 158 resolve
    identically in BOTH this repository and the sibling checkout, because
    both contain crates/onnx-genai-server/src/... at the same paths. So a
    bare "OK - every anchored citation resolves" is true of two different
    trees that disagree, and the tail of this output is the part people
    paste. A qualifier that is not adjacent to the claim is not attached
    to it.
    """
    try:
        root = tree_context.repo_root()
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        return f"{root.name} @ {branch}"
    except Exception:
        # Never let a cosmetic qualifier change a verdict. An unknown tree
        # is reported as unknown, not as an error and not as silence.
        return "UNKNOWN TREE"


def report(failures: list[Failure], stats: dict, doc: Path) -> int:
    print(f"citations in {doc}: anchored {stats['anchored']} | "
          f"positional {stats['unanchored']} | doc-mentions {stats['doc_mentions']} "
          f"| line-anchored continuations {stats['continuations']}")
    if stats.get("unanchored"):
        census = container_census(doc)
        if census["blockquoted"] or census["fenced"]:
            print(
                f"  CONTAINER CENSUS of those positional citations: "
                f"prose {census['prose']} | blockquoted {census['blockquoted']} | "
                f"fenced {census['fenced']}"
            )
            # The predicate travels WITH the number. These counts are not
            # comparable to any other blockquote count on this branch unless
            # that one uses the same three rules, and at least one independent
            # census used a different set and got a different total. A bare
            # number invites exactly the false disagreement this line prevents.
            print(
                "    (counted per OCCURRENCE, not per line; requires INLINE-CODE "
                "BACKTICKS; extensions rs/py/js/css/html/toml/md/sh)"
            )
        if census["blockquoted"]:
            # The ratchet tells people to re-anchor positional citations. For a
            # blockquoted one that order may have NO LEGAL COMPLIANCE PATH: if
            # the line quotes someone else's finding, editing the number
            # falsifies the quotation. A guard that can issue an unfollowable
            # order must say so at the moment it issues it, not in a doc.
            print(
                f"  WARNING - {census['blockquoted']} of them are inside BLOCKQUOTES. "
                f"This harness cannot tell a citation from a QUOTATION of someone "
                f"else's citation. If a quoted line is being re-anchored, editing "
                f"the number FALSIFIES THE QUOTE -- leave it and anchor the "
                f"surrounding prose instead. Do NOT 'fix' a quotation to satisfy "
                f"this tool."
            )
    pending = stats.get("reports") or []
    if pending:
        # Printed BEFORE the verdict, and printed even when the run is green.
        # A known-unchecked citation that is never mentioned becomes an
        # unknown-unchecked one within a day.
        #
        # GROUPED BY KIND, because a single header over two different defect
        # classes is the same fault this harness exists to catch: a label that
        # misdescribes what it covers. These two are not variants of one
        # problem -- a deictic citation is UNCHECKABLE, an ambiguous symbol is
        # CHECKED AND STILL INSUFFICIENT -- and they need opposite remedies.
        groups: dict[str, list[Failure]] = {}
        for f in pending:
            groups.setdefault(f.kind, []).append(f)
        headers = {
            "CONTINUATION_UNATTRIBUTABLE": "deictic citations that inherit their file from neighbouring "
                       "prose -- NOT content-checked at all (frozen by ratchet)",
            "AMBIGUOUS_SYMBOL": "citations whose symbol name is defined in more than one "
                                "tracked file -- each RESOLVES CORRECTLY, and the prose "
                                "beside it can still name the wrong owner",
        }
        for kind, items in sorted(groups.items()):
            print(f"\n--- {len(items)} non-fatal [{kind}]: "
                  f"{headers.get(kind, 'see detail')} ---")
            for f in items[:5]:
                print(f"  {f.citation.doc}:{f.citation.line_no}  {f.citation.raw}")
                if kind == "AMBIGUOUS_SYMBOL":
                    print(f"      {f.detail}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
    diverged = stats.get("divergence") or []
    if diverged:
        # BEFORE the verdict, and it applies to a red run exactly as much as a
        # green one. A failure measured against somebody else's uncommitted
        # draft is not a finding either.
        print(f"\n--- {len(diverged)} [WORKTREE_DIVERGENCE]: cited sources whose bytes "
              f"are NOT the bytes we ship ---")
        for line in diverged[:8]:
            print(f"  {line}")
        if len(diverged) > 8:
            print(f"  ... and {len(diverged) - 8} more")
    if not failures:
        print(f"OK [{tree_qualifier()}] - every anchored citation resolves to a "
              f"definition that exists IN THAT TREE. The sibling checkout has "
              f"files at identical paths; most citations resolve there too, so "
              f"this OK does NOT identify which repository the document means.")
        if diverged:
            print(f"WARNING - this OK was computed from the WORKING TREE, and "
                  f"{len(diverged)} cited source file(s) above differ from HEAD. "
                  f"It is a statement about one desk at one moment, NOT about the "
                  f"branch. Re-run once those files are committed before quoting it.")
        if pending:
            print(f"NOTE - this OK covers {stats['anchored']} anchored citations, and it "
                  f"means ONLY that each pointer lands on something real. It does NOT "
                  f"mean the surrounding prose is correct: {len(pending)} citations above "
                  f"are counted and NOT content-checked.")
        return 0
    by_kind: dict[str, list[Failure]] = {}
    for f in failures:
        by_kind.setdefault(f.kind, []).append(f)
    print(f"\nFAILURES BELOW ARE RELATIVE TO [{tree_qualifier()}]. A citation that "
          f"is wrong here may be correct in the sibling checkout, and vice versa.")
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

    # 8. A positional citation naming an UNTRACKED path must be caught. This
    #    category was previously counted and ratcheted but never resolved, so a
    #    citation the tool could not follow was tallied as one it had checked.
    def positional_unresolvable(repo: Path):
        (repo / "docs" / "d.md").write_text(
            SAMPLE_DOC + "Also see `engine/src/sample.rs:44` for detail.\n"
        )
    run_case(
        "positional citation to an untracked path",
        positional_unresolvable,
        "UNANCHORED_UNRESOLVABLE",
    )

    # 9. ...but a positional citation whose path IS tracked must NOT raise
    #    UNANCHORED_UNRESOLVABLE. Without this the check above would pass by
    #    flagging every positional citation indiscriminately, which proves
    #    nothing about whether it can actually resolve a path.
    def positional_resolvable(repo: Path):
        (repo / "docs" / "d.md").write_text(
            SAMPLE_DOC + "Also see `src/sample.rs:2` for detail.\n"
        )
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "src").mkdir()
        (repo / "docs").mkdir()
        (repo / "src" / "sample.rs").write_text(SAMPLE_SRC)
        (repo / "docs" / "d.md").write_text(SAMPLE_DOC)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "src/sample.rs", "docs/d.md"], cwd=repo, check=True)
        positional_resolvable(repo)
        failures, _ = check(repo, repo / "docs" / "d.md", {})
        kinds = {f.kind for f in failures}
        ok = "UNANCHORED_UNRESOLVABLE" not in kinds
        results.append((
            "positional citation to a TRACKED path (must not raise)",
            ok,
            "did not raise" if ok else f"falsely raised: {kinds}",
        ))

    # 10. A continuation citation naming a line beyond the end of the file it
    #     inherits is a provably broken citation and stays FATAL.
    def continuation_out_of_range(repo: Path):
        (repo / "docs" / "d.md").write_text(
            "See `src/sample.rs::cited_symbol` and also `:999` for the mapping.\n"
        )
    run_case(
        "continuation citation past end of inherited file",
        continuation_out_of_range,
        "CONTINUATION_OUT_OF_RANGE",
    )

    # 11. The negative control that makes case 10 mean something: an inherited,
    #     in-range continuation must NOT raise. Without this, case 10 could pass
    #     by flagging every continuation regardless of the line it names.
    def continuation_in_range(repo: Path):
        (repo / "docs" / "d.md").write_text(
            "See `src/sample.rs::cited_symbol` and also `:2` for the mapping.\n"
        )
    run_case("continuation citation within the inherited file", continuation_in_range, None)

    # 12. Anti-shrink: deleting all citations must NOT be a way to go green.
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
    # 13-15. WORKTREE_DIVERGENCE. These run against COMMITTED throwaway repos
    #     because the property under test is precisely "does the desk differ
    #     from what is committed", and a repo with no commit at all cannot
    #     express it. Every earlier case above skips the commit, which is fine
    #     for them and would silently make all three of these vacuous.
    def divergence_case(name: str, mutate, expect: set[str]):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "src").mkdir()
            (repo / "docs").mkdir()
            (repo / "src" / "sample.rs").write_text(SAMPLE_SRC)
            # Tracked, committed, and NEVER cited by the document. This file is
            # the entire point of case 15.
            (repo / "src" / "unrelated.rs").write_text("fn untouched_by_docs() {}\n")
            (repo / "docs" / "d.md").write_text(SAMPLE_DOC)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "-m", "base"], cwd=repo, check=True,
            )
            mutate(repo)
            _, stats = check(repo, repo / "docs" / "d.md", {})
            got = {
                line.split()[1].rstrip(":")
                for line in stats.get("divergence") or []
            }
            ok = got == expect
            results.append((
                name, ok,
                f"reported {got or 'nothing'}" if ok
                else f"expected {expect or 'nothing'}, reported {got or 'nothing'}",
            ))

    # 13. Control. A committed, untouched tree must report NO divergence, or
    #     cases 14 and 15 are both unreadable.
    divergence_case("divergence: clean committed tree (must be silent)",
                    lambda r: None, set())

    # 14. The cited source is edited on the desk. MUST be disclosed -- this is
    #     the live defect: a citation verdict computed from bytes nobody ships.
    def dirty_cited(repo: Path):
        p = repo / "src" / "sample.rs"
        p.write_text(p.read_text() + "\n// edited on one desk only\n")
    divergence_case("divergence: CITED source dirty (must disclose)",
                    dirty_cited, {"src/sample.rs"})

    # 15. THE CASE THAT DECIDES WHETHER THIS INSTRUMENT IS WORTH ANYTHING.
    #     A tracked file that the document does not cite is edited. It must be
    #     SILENT. Without this arm, an implementation that simply printed
    #     `git status` would pass case 14 perfectly -- and would then cry wolf
    #     on every unrelated edit on a fourteen-agent branch until somebody
    #     switched it off. The claim is not "the tree is dirty". It is "the
    #     bytes THIS VERDICT RESTS ON are not the bytes we ship", and only a
    #     failing case 15 can tell those two apart.
    def dirty_uncited(repo: Path):
        p = repo / "src" / "unrelated.rs"
        p.write_text(p.read_text() + "\n// noise from another agent\n")
    divergence_case("divergence: UNCITED source dirty (must stay silent)",
                    dirty_uncited, set())

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
    ap.add_argument(
        "--require-branch",
        help="fail unless the tree being checked is on this branch",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.doc:
        ap.error("a document is required (or --self-test)")

    try:
        repo = tree_context.repo_root()
    except tree_context.NoWorktree as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        sys.exit(tree_context.CANNOT_RUN)
    ctx = tree_context.tree_context(repo)
    print(tree_context.banner(ctx), file=sys.stderr)
    if args.require_branch:
        try:
            tree_context.require_branch(args.require_branch, ctx)
        except tree_context.WrongTree as e:
            print(f"WRONG_TREE: {e}", file=sys.stderr)
            return 1
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
