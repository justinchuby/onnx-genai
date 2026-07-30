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

# Anti-shrink floor. A provenance table cannot get more honest by getting
# smaller: an entry that disappears takes its "this field is not measured"
# warning with it, and the field it described becomes an unannotated number.
#
# This floor is not hypothetical. While correcting this very table I ran a
# regex edit that silently deleted five entries, and every content check still
# passed green -- the surviving quotations were all still accurate. Correctness
# of what remains says nothing about what left. Raise this floor when entries
# are added; do not lower it to make a red go away.
MIN_ENTRIES = 36

# Vacuity floor. A checker that finds nothing to check must fail, not pass.
#
# This is not a hypothetical either: breaking the quote matcher so it matched
# nothing made this script print "0 quotations verified against source" and
# then "all provenance quotations still occur in the source they cite" and exit
# 0. Both sentences were true and the conjunction was worthless. A checker with
# no input is indistinguishable from a checker with nothing wrong, and it gets
# MORE trusted with every green run because nobody re-reads a passing tool.
#
# We have spent this session asking whether checks catch a bad value and never
# whether they notice they have nothing to look at.
MIN_QUOTES_CHECKED = 8

# Entries whose absence would be most expensive: each one annotates a field
# that a panel renders, so losing the entry silently promotes an unmeasured
# number to an unlabelled one.
REQUIRED_KEYS = (
    "batch.utilization",
    "queue.depth",
    "kv.usage",
    "kv.pages_used",
    "kv.pages_total",
    "kv.pages_shared",
    "sessions.paused",
    "throughput.tokens_per_second",
    "prefix_cache.hashes",
)

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

    if len(entries) < MIN_ENTRIES:
        failures.append(
            f"provenance table has {len(entries)} entries, floor is {MIN_ENTRIES}. "
            f"Entries were removed. A field with no provenance entry renders as a "
            f"number with no honesty label; fix by restoring the entries, not by "
            f"lowering MIN_ENTRIES."
        )
    for key in REQUIRED_KEYS:
        if key not in entries:
            failures.append(f"required provenance entry '{key}' is missing from the table")

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

        # A `reason` is shown to a VISITOR to explain why a number is absent, so
        # it is an assertion about the system exactly as much as the evidence is
        # -- and it is the least audited prose we produce, because a reader
        # treats it as an apology rather than as a claim. A wrong number gets
        # challenged; a wrong explanation gets sympathy. So any source text it
        # quotes is held to the identical standard.
        for field_name in ("evidence", "reason"):
            text = e[field_name]
            for quote in BACKTICKED.findall(text):
                q = normalise(quote)
                # Skip bare identifiers: they are prose references, not
                # quotations of a source line, and demanding they appear
                # verbatim would flag correct English. A phrase is not a claim.
                if len(q) < 8 or not re.search(r"[:(=]", q):
                    continue
                stats["quotes_checked"] += 1
                if q not in haystack:
                    failures.append(
                        f"{key}: the {field_name} quotes `{quote}` as coming from "
                        f"{rel}, but that text is no longer in the file. The record "
                        f"may be describing behaviour the source no longer has."
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

    if stats["quotes_checked"] < MIN_QUOTES_CHECKED:
        failures.append(
            f"only {stats['quotes_checked']} quotation(s) were checked, floor is "
            f"{MIN_QUOTES_CHECKED}. The checker found almost nothing to verify, "
            f"which means its matcher is broken or the evidence stopped quoting "
            f"source -- not that the table is correct. A check with no input is "
            f"indistinguishable from a check with nothing wrong."
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
