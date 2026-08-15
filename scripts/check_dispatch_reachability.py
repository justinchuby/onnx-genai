#!/usr/bin/env python3
"""Check that every dispatch branch in the CPU EP has a reachability test.

The rule
--------
Every ``static ...TEST_HITS: AtomicUsize`` counter in the CPU EP kernel code
must be read (via ``.load(...)``) inside a ``#[test]`` function in the same
file. A counter with no test proves nothing — the dispatch path it instruments
is *claimed* to be covered but actually never exercised in CI.

Conversely, every ``fetch_add`` to a ``TEST_HITS`` counter (the dispatch-side
instrumentation) must have a corresponding ``static`` declaration — a stray
fetch_add with no declared counter would be a build error, but this lint checks
for it explicitly as a coherence guarantee.

Why this rule exists
--------------------
PR #275 shipped two silent-wrong-answer bugs that passed codecov with 78% line
coverage. The rescue block at ``matmul.rs:758-830`` returned all zeros for
non-constant non-contiguous f16 B at M≥2. Every test passed
``constant_inputs = [false, true]`` with contiguous B — no test exercised the
actual bug path.

Line coverage cannot detect this class of defect. A reachability counter
asserts the property that actually matters: **this specific dispatch path
executes in the configuration we claim it does.**

The dispatch-reachability pattern (atomic hit counters) has since caught two
real regressions:
  - ``half_gemm.rs`` intercepting M=1 before the bandwidth-optimal GEMV
    (4.5× decode throughput regression)
  - Non-contiguous rescue block entering without ``constant_inputs[1]`` guard
    (silent all-zeros output)

Known gaps
----------
This lint enforces **counter ↔ test** pairing. It does NOT detect:

1. A dispatch branch that *should* have a counter but doesn't. That requires
   human review at PR time (the rule is: "any new dispatch branch ships with a
   ``_TEST_HITS`` counter"). This is a process gap, not a tooling gap — the
   lint cannot infer which ``if`` blocks are "dispatch branches" vs. ordinary
   control flow. We document this explicitly rather than pretend the lint is
   complete.

2. A test that reads the counter but makes a weak assertion (e.g. testing only
   that the counter is non-negative). Human review must verify the assertion is
   ``after > before`` (proves the path ran) or ``after == before`` (proves the
   path did NOT run).

3. Counters in ``#[cfg(test)]`` blocks that are dead on non-test builds. This
   is by design — the counters exist only for test verification and are
   compiled out of release builds.

The **one gap worth stating plainly**: this lint cannot catch a *missing*
counter on a new dispatch branch. That is a review-time responsibility.
``scripts/check_platform_naming.py`` explicitly states the same class of
limitation for within-file gaps. Our layered defense is:
  - Platform naming lint: catches file-level single-arch code without fallback
  - Dispatch reachability lint (this script): catches counter-without-test
  - Human review: catches branch-without-counter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "crates" / "onnx-runtime-ep-cpu" / "src" / "kernels"

# Matches: static SOMETHING_TEST_HITS: ... AtomicUsize/AtomicU64 ...
COUNTER_DECL_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?static\s+(\w+TEST_HITS)\s*:\s*"
    r"(?:std::sync::atomic::)?Atomic(?:Usize|U64)",
    re.MULTILINE,
)

# Matches: COUNTER_NAME.load(...)  (reading the counter in a test)
def counter_load_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\.load\s*\(")

# Matches: COUNTER_NAME.fetch_add(...)  (incrementing in dispatch code)
def counter_fetch_re(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\.fetch_add\s*\(")

# Matches the start of a #[cfg(test)] module or a #[test] function
TEST_BLOCK_RE = re.compile(r"#\[cfg\(test\)\]|#\[test\]")


def find_test_region_lines(text: str) -> set[int]:
    """Return line numbers that are inside a #[cfg(test)] module or a #[test] fn."""
    lines = text.split("\n")
    in_test = False
    brace_depth = 0
    test_lines: set[int] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_test:
            if "#[cfg(test)]" in stripped or "#[test]" in stripped:
                in_test = True
                brace_depth = 0
                test_lines.add(i)
                continue
        else:
            test_lines.add(i)
            brace_depth += line.count("{") - line.count("}")
            # End of test region when we close back to or below zero
            if brace_depth < 0:
                in_test = False

    return test_lines


def check_file(path: Path) -> list[str]:
    """Check one .rs file for counter/test pairing violations."""
    text = path.read_text()
    problems: list[str] = []

    # Find all counter declarations
    counters = COUNTER_DECL_RE.findall(text)
    if not counters:
        return []

    rel = path.relative_to(REPO)

    # Find test regions (lines inside #[cfg(test)] or #[test])
    test_lines = find_test_region_lines(text)
    lines = text.split("\n")

    for counter in counters:
        # Check: is this counter ever .load()'d inside a test?
        # Strip line comments before matching to avoid false positives from
        # commented-out code.
        load_pat = counter_load_re(counter)
        has_test_read = False
        for i, line in enumerate(lines):
            if i in test_lines:
                # Strip // comment (naive but sufficient for this pattern)
                code_part = line.split("//")[0]
                if load_pat.search(code_part):
                    has_test_read = True
                    break

        if not has_test_read:
            # Find the declaration line for the error message
            for i, line in enumerate(lines):
                if re.search(rf"^(?:pub(?:\([^)]*\))?\s+)?static\s+{re.escape(counter)}\b", line):
                    decl_line = i + 1
                    break
            else:
                decl_line = 0

            problems.append(
                f"  {rel}:{decl_line}: `{counter}` has no test reading it\n"
                f"\n"
                f"    A dispatch-reachability counter without a test proves nothing.\n"
                f"    The counter instruments a dispatch branch, but no #[test]\n"
                f"    function asserts that it was incremented (or not incremented).\n"
                f"\n"
                f"    This is the exact pattern that let PR #275 ship two silent-\n"
                f"    wrong-answer bugs: the non-contiguous f16 rescue block at\n"
                f"    matmul.rs:758-830 returned all zeros for non-constant B, while\n"
                f"    every test used contiguous constant B and passed.\n"
                f"\n"
                f"    Fix: add a #[test] that calls .load() on `{counter}` before\n"
                f"    and after executing the dispatch path, and asserts the\n"
                f"    expected delta (after > before for reachability, after ==\n"
                f"    before for exclusion)."
            )

    # Also check: are there fetch_add calls to names that aren't declared?
    fetch_re = re.compile(r"\b(\w+TEST_HITS)\.fetch_add\s*\(")
    all_fetch_names = set(fetch_re.findall(text))
    counter_set = set(counters)
    orphan_fetches = all_fetch_names - counter_set
    for name in sorted(orphan_fetches):
        problems.append(
            f"  {rel}: `{name}.fetch_add(...)` found but no matching\n"
            f"    `static {name}: AtomicUsize` declaration in this file.\n"
            f"    Either declare the counter or remove the stale instrumentation."
        )

    return problems


def main() -> int:
    if not KERNELS.is_dir():
        sys.exit(f"kernel directory not found: {KERNELS}")

    all_problems: list[str] = []
    scanned = 0
    counters_found = 0

    for rs_file in sorted(KERNELS.rglob("*.rs")):
        scanned += 1
        problems = check_file(rs_file)
        if problems:
            # Count counters in this file for reporting
            text = rs_file.read_text()
            counters_found += len(COUNTER_DECL_RE.findall(text))
            all_problems.extend(problems)
        else:
            text = rs_file.read_text()
            counters_found += len(COUNTER_DECL_RE.findall(text))

    if all_problems:
        print(
            "dispatch-reachability lint: every _TEST_HITS counter must have a\n"
            "corresponding #[test] that reads it:\n"
        )
        for p in all_problems:
            print(p)
        print(
            f"\nScanned {scanned} file(s), found {counters_found} counter(s).\n"
            f"\n"
            f"The dispatch-reachability pattern requires:\n"
            f"  1. A `static FOO_TEST_HITS: AtomicUsize` in the dispatch code\n"
            f"  2. A `FOO_TEST_HITS.fetch_add(1, ...)` at the branch entry point\n"
            f"  3. A #[test] that reads the counter with `.load()` and asserts\n"
            f"     `after > before` (path reached) or `after == before` (excluded)\n"
            f"\n"
            f"Known gap: this lint cannot detect a dispatch branch that SHOULD\n"
            f"have a counter but doesn't. That is a review-time responsibility.\n"
            f"See: scripts/check_dispatch_reachability.py docstring for details."
        )
        return 1

    print(
        f"dispatch-reachability lint: {scanned} file(s) checked, "
        f"{counters_found} counter(s) all paired with tests ✓"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
