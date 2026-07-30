#!/usr/bin/env python3
"""Verify every quoted source claim in telemetry-provenance.js against the source.

WHY THIS EXISTS, AND WHY IT IS NOT THE SAME CHECK AS check_citations.py
----------------------------------------------------------------------
check_citations.py answers "does this citation still point at the symbol it
names?". That is a question about a *location*. This script answers a different
and, for the provenance table, more dangerous question: "does the source still
DO what this record says it does?"

The provenance table feeds the demo footer that tells a skeptical visitor which
numbers are real. It is the one artefact in the project whose entire value is
that it is trusted. So its failure mode is inverted from everywhere else:

  * everywhere else, the expensive defect is a FAKE VALUE described as real
  * here, the expensive defect is a REAL VALUE described as a fabrication

The second kind cannot be caught by any check that looks for hardcoded zeros,
because there is no zero to find. It is caught only by comparing the quoted
text against the file it is quoted from. That is what this does.

Both directions fail silently in the product:
  - "fake" over a real field  -> the footer denies a panel that is working, and
    if the store gates display on provenance the panel renders em-dashes over
    live telemetry.
  - "real" over a stub        -> the footer certifies a fabrication.

MECHANISM: content-anchored, never line-anchored. A provenance entry cites
source by quoting it in backticks. We require that quoted text to still occur
in the cited file. Line numbers are deliberately NOT checked and NOT required,
because a line number that drifts produces a false alarm while the claim is
still true -- and a checker that cries wolf gets its assertions loosened, which
is how a safeguard dies.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROVENANCE_JS = "examples/serving-dashboard/telemetry-provenance.js"

# A quoted source fragment inside an evidence string: `foo: None`
BACKTICKED = re.compile(r"`([^`]+)`")
# Path to a Rust/JS source file, with or without a trailing :line
SOURCE_REF = re.compile(r"((?:crates|examples|scripts)/[\w./-]+\.(?:rs|js|py))(?::(\d+))?")


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def load_entries(repo: Path) -> dict:
    """Read the provenance table by executing the module, not by parsing it.

    Parsing the JS with a regex would re-create the exact defect this script
    exists to catch: a second, drifting representation of the truth. Node is
    the only thing that agrees with the browser about what the table says.
    """
    script = (
        "import(process.argv[1]).then(m=>{"
        "const out={};"
        "for(const [k,e] of Object.entries(m.PROVENANCE)){"
        "out[k]={classification:e.classification,evidence:String(e.evidence||''),"
        "reason:String(e.reason||''),stubValue:e.stubValue};}"
        "process.stdout.write(JSON.stringify(out));})"
    )
    res = subprocess.run(
        ["node", "-e", script, str(repo / PROVENANCE_JS)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"FATAL: could not load {PROVENANCE_JS}\n{res.stderr}", file=sys.stderr)
        sys.exit(2)
    return json.loads(res.stdout)


def normalise(s: str) -> str:
    """Collapse whitespace so a quote survives rustfmt rewrapping a line.

    Rust string literals are frequently split across lines with a trailing
    backslash. Comparing raw text would go red on a pure formatting change,
    which is a false alarm, and false alarms are how checks get deleted.
    """
    s = s.replace("\\\n", "")
    return re.sub(r"\s+", " ", s).strip()


def check(repo: Path) -> tuple[list[str], dict]:
    entries = load_entries(repo)
    failures: list[str] = []
    stats = {"entries": len(entries), "quotes_checked": 0, "entries_with_source": 0}
    cache: dict[str, str] = {}

    for key, e in sorted(entries.items()):
        evidence = e["evidence"]
        m = SOURCE_REF.search(evidence)
        if not m:
            continue
        stats["entries_with_source"] += 1
        rel = m.group(1)
        path = repo / rel
        if not path.exists():
            failures.append(f"{key}: cites {rel}, which does not exist")
            continue
        if rel not in cache:
            cache[rel] = normalise(path.read_text(errors="replace"))
        haystack = cache[rel]

        for quote in BACKTICKED.findall(evidence):
            q = normalise(quote)
            # Skip bare identifiers: they are prose references, not quotations
            # of a source line, and demanding they appear verbatim would flag
            # correct English. A phrase is not a claim.
            if len(q) < 8 or not re.search(r"[:(=]", q):
                continue
            stats["quotes_checked"] += 1
            if q not in haystack:
                failures.append(
                    f"{key}: provenance quotes `{quote}` as coming from {rel}, "
                    f"but that text is no longer in the file. The record may be "
                    f"describing behaviour the source no longer has."
                )

        # Directional guard: an entry classified as never-measured must not be
        # describing a field the source now computes for real. This is the
        # expensive direction -- a real value under a record calling it fake.
        if e["classification"] in ("DOCUMENTED_ZERO", "NOT_PLUMBED", "STRUCTURALLY_BYPASSED"):
            field = re.search(r"`(\w+):\s*([^`,]+)", evidence)
            if field and field.group(2).strip() not in ("None", "0", "0.0"):
                failures.append(
                    f"{key}: classified {e['classification']} (never measured) but its own "
                    f"evidence quotes a non-stub value `{field.group(1)}: {field.group(2)}`"
                )

    return failures, stats


def main() -> int:
    repo = repo_root()
    failures, stats = check(repo)
    print(
        f"provenance: {stats['entries']} entries, "
        f"{stats['entries_with_source']} citing source, "
        f"{stats['quotes_checked']} quotations verified against source"
    )
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all provenance quotations still occur in the source they cite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
