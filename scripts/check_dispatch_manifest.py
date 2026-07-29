#!/usr/bin/env python3
"""Check that every claimed optimization in dispatch_manifest.toml is backed
by a reachability counter in the declared source file.

The rule
--------
Each ``[[claim]]`` row in ``dispatch_manifest.toml`` declares that on a given
platform, a given op dispatches to at least a specified tier, and that a named
``_TEST_HITS`` counter proves it.  This lint verifies:

  1. The ``file`` path exists.
  2. The ``counter`` is declared as a ``static ...: AtomicUsize`` in that file.
  3. (Informational) The counter is incremented somewhere (``fetch_add``).

If any claim is unsatisfied, the lint fails with a message naming the op,
platform, expected tier, counter, and what is missing.

Why this exists
---------------
``check_dispatch_reachability.py`` enforces that every counter has a test.
But it cannot detect a *missing* counter — if nobody creates one, that check
passes happily.  This was exploited by instances #8 and #9 of the structural
bug: ``conv_ref.rs`` had no BNNS counter, and the reachability lint had nothing
to flag because there was nothing to pair.

The manifest closes the loop: **if we claim an optimization exists, the counter
that proves it must exist too.**  A claim without a counter fails CI at PR time
— before the silent-fallback has a chance to ship.

Relationship to other lints
---------------------------
  check_platform_naming.py   → file-level: single-arch file without name marker
  check_dispatch_reachability.py → counter-level: counter without paired test
  check_dispatch_manifest.py → claim-level: optimization without proving counter

Each covers what the others cannot.  Together they form a layered defense.

Cross-EP extensibility
----------------------
The manifest format is EP-agnostic.  The ``file`` field can point anywhere in
the workspace.  A CUDA EP optimization would declare its counter in a CUDA
kernel file; the lint validates it identically.  No EP-specific logic exists in
this script — only "does this file contain this counter?"

Today all claims are CPU EP because that is where counters exist.  As Metal/
CUDA/plugin EPs adopt the TEST_HITS pattern, they add rows and get the same
protection for free.

Known gaps
----------
1. This lint does NOT verify that the counter is reached at test time.  That is
   the job of the test itself (which ``check_dispatch_reachability.py`` ensures
   exists).  The layering is: manifest → counter exists → counter has test →
   test asserts counter increments.

2. The manifest cannot invent claims.  If nobody adds a row for a new
   optimization, it ships unguarded.  **Mitigated** by the inverse check
   (added in this version): any _TEST_HITS counter whose name does NOT
   contain SCALAR/FALLBACK/RESCUE/REF is flagged if it lacks a manifest row.
   This catches the exact case where PR #324 shipped three optimized paths
   without manifest rows within an hour of the manifest being created.

3. The ``platform`` field is not validated against actual cfg conditions in
   source.  A row claiming ``aarch64-apple-darwin`` is not verified to be
   inside a ``cfg(target_os = "macos", target_arch = "aarch64")`` block.  The
   test itself (running on CI macOS runner) provides that platform assertion.

4. Tier ordering is not numerically enforced across claims — the manifest
   records "minimum_tier" per row independently.  If tier definitions change,
   human review must update rows.  The vocabulary is intentionally minimal
   (tier1/tier2/tier3) to resist churn.

5. Graph-level optimizations (e.g. Conv+BatchNorm fusion) that eliminate an op
   entirely cannot be expressed as a dispatch tier claim.  They require a
   separate enforcement mechanism (optimizer-level counters).  The manifest
   documents these honestly as [[exclusion]] rows rather than misrepresenting
   them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        # Inline fallback: parse just enough TOML for our manifest format.
        # This avoids adding a pip dependency to CI.
        tomllib = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "dispatch_manifest.toml"

# Matches: static COUNTER_NAME: ... AtomicUsize ...
COUNTER_DECL_RE = re.compile(
    r"static\s+{name}\s*:\s*"
    r"(?:std::sync::atomic::)?AtomicUsize",
)

# Matches: COUNTER_NAME.fetch_add(
COUNTER_FETCH_RE = re.compile(r"{name}\.fetch_add\s*\(")

VALID_TIERS = {"tier1", "tier2", "tier3"}


def parse_manifest_fallback(text: str) -> dict:
    """Minimal TOML parser for the dispatch manifest (no external deps).

    Handles only the subset we use: [[claim]] and [[exclusion]] arrays of
    tables with string values.  This is NOT a general TOML parser.
    """
    result: dict[str, list[dict[str, str]]] = {"claim": [], "exclusion": []}
    current_table: str | None = None
    current_entry: dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Array-of-tables header
        header_match = re.match(r"^\[\[(\w+)\]\]$", stripped)
        if header_match:
            # Save previous entry
            if current_table and current_entry:
                result.setdefault(current_table, []).append(current_entry)
            current_table = header_match.group(1)
            current_entry = {}
            continue

        # Key = "value" or key = 'value'
        kv_match = re.match(r'^(\w+)\s*=\s*"([^"]*)"$', stripped)
        if not kv_match:
            kv_match = re.match(r"^(\w+)\s*=\s*'([^']*)'$", stripped)
        if kv_match and current_table:
            current_entry[kv_match.group(1)] = kv_match.group(2)

    # Save last entry
    if current_table and current_entry:
        result.setdefault(current_table, []).append(current_entry)

    return result


def load_manifest() -> dict:
    """Load and parse the dispatch manifest."""
    if not MANIFEST_PATH.exists():
        sys.exit(
            f"dispatch-manifest lint: manifest not found at {MANIFEST_PATH}\n"
            f"  Expected: dispatch_manifest.toml in the repository root."
        )

    text = MANIFEST_PATH.read_text()

    if tomllib is not None:
        return tomllib.loads(text)
    else:
        return parse_manifest_fallback(text)


def check_counter_in_file(
    counter: str, file_path: Path
) -> tuple[bool, bool, str]:
    """Check if a counter is declared and incremented in the given file.

    Returns (declared, incremented, detail_message).
    """
    if not file_path.exists():
        return False, False, f"file does not exist: {file_path.relative_to(REPO)}"

    text = file_path.read_text()

    decl_re = re.compile(
        rf"(?:pub(?:\([^)]*\))?\s+)?static\s+{re.escape(counter)}\s*:\s*"
        rf"(?:std::sync::atomic::)?Atomic(?:Usize|U64)"
    )
    declared = bool(decl_re.search(text))

    # `\s*` around the dot is load-bearing, not defensive. rustfmt wraps a long
    # increment as
    #     COUNTER
    #         .fetch_add(1, Ordering::Relaxed);
    # and a `COUNTER\.fetch_add` pattern then reports a live branch as dead.
    # That happened to CONV_POINTWISE_GEMM_TEST_HITS: the lint called a working
    # 1x1-pointwise dispatch "likely dead code" purely because the line was long.
    # A lint that cries wolf over formatting teaches people to ignore it, which
    # costs more than the check is worth. See self_test().
    fetch_re = re.compile(rf"\b{re.escape(counter)}\s*\.\s*fetch_add\s*\(")
    incremented = bool(fetch_re.search(text))

    if not declared:
        return False, False, (
            f"counter `{counter}` not found as `static {counter}: Atomic{{Usize,U64}}` "
            f"in {file_path.relative_to(REPO)}"
        )

    if not incremented:
        return True, False, (
            f"counter `{counter}` is declared but never incremented "
            f"(no .fetch_add()) in {file_path.relative_to(REPO)}"
        )

    return True, True, ""


def main() -> int:
    manifest = load_manifest()
    claims = manifest.get("claim", [])

    if not claims:
        print(
            "dispatch-manifest lint: no [[claim]] entries found in manifest.\n"
            "  The manifest exists but declares no optimizations.\n"
            "  This is valid but unusual — add claims as optimizations ship."
        )
        return 0

    problems: list[str] = []
    warnings: list[str] = []
    checked = 0

    for claim in claims:
        # Validate required fields
        missing_fields = []
        for field in ("op", "platform", "minimum_tier", "counter", "file"):
            if field not in claim:
                missing_fields.append(field)

        if missing_fields:
            problems.append(
                f"  [[claim]] missing required field(s): {', '.join(missing_fields)}\n"
                f"    Present fields: {claim}\n"
                f"\n"
                f"    Every [[claim]] row needs: op, platform, minimum_tier, counter, file"
            )
            continue

        op = claim["op"]
        variant = claim.get("variant", "default")
        platform = claim["platform"]
        tier = claim["minimum_tier"]
        counter = claim["counter"]
        file_rel = claim["file"]
        desc = claim.get("description", "(no description)")

        # Validate tier value
        if tier not in VALID_TIERS:
            problems.append(
                f"  {op}/{variant} on {platform}: invalid minimum_tier '{tier}'\n"
                f"    Valid values: {', '.join(sorted(VALID_TIERS))}"
            )
            continue

        # Check counter exists in the declared file
        file_path = REPO / file_rel
        declared, incremented, detail = check_counter_in_file(counter, file_path)
        checked += 1

        if not declared:
            problems.append(
                f"  {op}/{variant} on {platform} — CLAIM UNSATISFIED\n"
                f"    Expected tier: {tier}\n"
                f"    Counter: {counter}\n"
                f"    File: {file_rel}\n"
                f"    Problem: {detail}\n"
                f"    Description: {desc}\n"
                f"\n"
                f"    This means the optimization is CLAIMED but not INSTRUMENTED.\n"
                f"    Either:\n"
                f"      a) Add `static {counter}: AtomicUsize` to the dispatch path, or\n"
                f"      b) Remove the [[claim]] row if the optimization was reverted."
            )
        elif not incremented:
            warnings.append(
                f"  {op}/{variant} on {platform} — counter declared but never incremented\n"
                f"    Counter: {counter} in {file_rel}\n"
                f"    This likely means the dispatch branch is dead code.\n"
                f"    Description: {desc}"
            )

    # Report
    if problems:
        print(
            "dispatch-manifest lint FAILED: claimed optimizations lack proving "
            "counters:\n"
        )
        for p in problems:
            print(p)
            print()

        if warnings:
            print("Additionally, these counters are declared but never fire:\n")
            for w in warnings:
                print(w)
                print()

        print(
            f"Checked {checked} claim(s) in {MANIFEST_PATH.relative_to(REPO)}.\n"
            f"\n"
            f"WHY THIS CHECK EXISTS:\n"
            f"  The dispatch manifest declares which optimizations we CLAIM to\n"
            f"  deliver on each platform. Each claim must be backed by a\n"
            f"  _TEST_HITS counter that a test reads (enforced by\n"
            f"  check_dispatch_reachability.py). Without this pairing:\n"
            f"    - A claimed optimization can silently fall to scalar reference\n"
            f"    - CI passes because correctness tests don't test speed\n"
            f"    - Only benchmarks notice, and only if someone runs them\n"
            f"\n"
            f"  This is the 'Conv on macOS' bug: 643× slower than ORT, green CI,\n"
            f"  because conv_ref.rs (scalar) had no counter proving BNNS ran.\n"
            f"\n"
            f"FIX:\n"
            f"  When shipping an optimized dispatch path:\n"
            f"    1. Add a `static MY_OP_TEST_HITS: AtomicUsize` counter\n"
            f"    2. Increment it at the dispatch branch entry: .fetch_add(1, ...)\n"
            f"    3. Add a #[test] that asserts the counter increments\n"
            f"    4. Add a [[claim]] row to dispatch_manifest.toml\n"
            f"\n"
            f"  Steps 1-3 are the existing pattern. Step 4 is new — it declares\n"
            f"  the optimization so that removing steps 1-3 becomes a CI failure\n"
            f"  rather than a silent regression."
        )
        return 1

    # All good
    if warnings:
        print("dispatch-manifest lint: all claims satisfied, with warnings:\n")
        for w in warnings:
            print(w)
        print()

    print(
        f"dispatch-manifest lint: {checked} claim(s) verified, "
        f"all counters present ✓"
    )

    # Also report exclusions for visibility
    exclusions = manifest.get("exclusion", [])
    if exclusions:
        print(f"\n  ({len(exclusions)} deliberate exclusion(s) documented)")

    # ─── Inverse check: counters without manifest rows ─────────────────────
    # A _TEST_HITS counter is a strong signal that someone added a dispatch
    # branch. If it's an *optimization* counter (not a scalar/fallback/rescue
    # counter) and has no manifest row, it means the optimization shipped
    # unguarded. This would have caught all three of PR #324's paths.
    #
    # False-positive filter: counters whose names contain SCALAR, FALLBACK,
    # RESCUE, or REF are proving fallback paths, not claiming optimizations.
    # Those do not need manifest rows.

    FALLBACK_MARKERS = {"SCALAR", "FALLBACK", "RESCUE", "REF"}

    claimed_counters = {c.get("counter") for c in claims if "counter" in c}

    # Scan kernel directories for all counters
    scan_dirs = [
        REPO / "crates" / "onnx-runtime-ep-cpu" / "src" / "kernels",
        REPO / "crates" / "onnx-runtime-session" / "src" / "executor",
    ]

    unclaimed: list[str] = []
    all_counter_re = re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?static\s+(\w+TEST_HITS)\s*:\s*"
        r"(?:std::sync::atomic::)?Atomic(?:Usize|U64)",
        re.MULTILINE,
    )

    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for rs_file in sorted(scan_dir.rglob("*.rs")):
            text = rs_file.read_text()
            for match in all_counter_re.finditer(text):
                counter_name = match.group(1)
                # Skip fallback/scalar counters — these are not optimization claims
                if any(marker in counter_name for marker in FALLBACK_MARKERS):
                    continue
                if counter_name not in claimed_counters:
                    rel = rs_file.relative_to(REPO)
                    unclaimed.append(
                        f"  {counter_name} in {rel}"
                    )

    if unclaimed:
        print(
            "\n"
            "dispatch-manifest lint FAILED: optimization counters exist without "
            "manifest claims:\n"
        )
        for u in unclaimed:
            print(u)
        print(
            f"\n"
            f"A _TEST_HITS counter that is not SCALAR/FALLBACK/RESCUE/REF\n"
            f"indicates an optimized dispatch path. Each such path should have\n"
            f"a [[claim]] row in dispatch_manifest.toml declaring what tier it\n"
            f"provides on which platform.\n"
            f"\n"
            f"Without a manifest row, the optimization is UNGUARDED: removing\n"
            f"the fast path silently falls to scalar reference with no CI failure.\n"
            f"\n"
            f"WHY THIS MATTERS:\n"
            f"  PR #324 shipped three new optimizations (MaxPool→BNNS, Add→vDSP,\n"
            f"  BatchNorm opset-7) one hour after the manifest was created. All\n"
            f"  three had counters but no manifest rows — exactly the condition\n"
            f"  that let prior defects survive for months.\n"
            f"\n"
            f"FIX: Add a [[claim]] row to dispatch_manifest.toml for each counter,\n"
            f"or if the counter proves a deliberate fallback path, rename it to\n"
            f"include SCALAR/FALLBACK/RESCUE/REF in the name."
        )
        return 1

    return 0


def self_test() -> int:
    """Check the lint's own detection regexes against realistic rustfmt output.

    This lint is a detector, and a detector's failure mode is silence. These
    cases exist because the counter/increment patterns are matched textually,
    so any formatting the regexes did not anticipate becomes a blind spot.
    Every case below is a shape rustfmt actually produces.
    """
    import tempfile

    cases = [
        # (label, source, expect_declared, expect_incremented)
        (
            "single-line increment",
            "static C_TEST_HITS: AtomicUsize = AtomicUsize::new(0);\n"
            "fn f() { C_TEST_HITS.fetch_add(1, Ordering::Relaxed); }\n",
            True,
            True,
        ),
        (
            "rustfmt-wrapped increment (the CONV_POINTWISE_GEMM regression)",
            "static C_TEST_HITS: AtomicUsize = AtomicUsize::new(0);\n"
            "fn f() {\n    C_TEST_HITS\n"
            "        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);\n}\n",
            True,
            True,
        ),
        (
            "rustfmt-wrapped declaration",
            "static C_TEST_HITS: std::sync::atomic::AtomicUsize =\n"
            "    std::sync::atomic::AtomicUsize::new(0);\n"
            "fn f() { C_TEST_HITS.fetch_add(1, Ordering::Relaxed); }\n",
            True,
            True,
        ),
        (
            "genuinely dead counter must still be reported",
            "static C_TEST_HITS: AtomicUsize = AtomicUsize::new(0);\n"
            "fn f() { let _ = C_TEST_HITS.load(Ordering::Relaxed); }\n",
            True,
            False,
        ),
        (
            "missing counter must still be reported",
            "fn f() {}\n",
            False,
            False,
        ),
    ]

    failures = []
    with tempfile.TemporaryDirectory(dir=REPO) as tmp:
        for label, src, want_decl, want_incr in cases:
            path = Path(tmp) / "case.rs"
            path.write_text(src)
            declared, incremented, _ = check_counter_in_file("C_TEST_HITS", path)
            if (declared, incremented) != (want_decl, want_incr):
                failures.append(
                    f"  {label}\n"
                    f"    expected declared={want_decl} incremented={want_incr}\n"
                    f"    got      declared={declared} incremented={incremented}"
                )

    if failures:
        print("dispatch-manifest lint SELF-TEST FAILED:\n")
        print("\n".join(failures))
        print(
            "\nThe lint's own detection is broken. Until this passes, a green\n"
            "manifest check proves nothing."
        )
        return 1

    print(f"dispatch-manifest lint self-test: {len(cases)} case(s) passed ✓")
    return 0


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
