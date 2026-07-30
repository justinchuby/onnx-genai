#!/usr/bin/env python3
"""Anchor the provenance table's classifications to what a server actually sends.

WHY THIS IS NOT check_provenance.py
-----------------------------------
check_provenance.py asks "does the source still DO what this record says?" -- it
compares quoted text against the file it was quoted from. That is a claim about
CODE.

A classification is a different claim. `NOT_PLUMBED` asserts that a field does
not arrive on the wire. `MEASURED` asserts that it does. Neither is settleable
from source, because whether a field is served depends on the endpoint contract,
the model, and the flags the process was launched with. The only artefact that
can settle it is a running server.

THE DIRECTION NOBODY CHECKS
---------------------------
Every other guard in this repository looks for an overclaim: a fabricated number
described as real. This one is bidirectional, and the half that matters is the
one no other instrument watches:

  * MEASURED over a field the server declares unavailable -> OVERCLAIM. The
    footer certifies a number the server never sent.
  * NOT_PLUMBED over a field the server really serves     -> UNDERCLAIM. The
    footer denies a working panel, and a document that understates our honesty
    is still a document that is wrong. Nobody ever greps to check whether we are
    being too hard on ourselves, so this rots silently and permanently.

WHY IT PARSES AND NEVER GREPS
-----------------------------
The obvious implementation is to search the payload text for the field name.
That is wrong, and it is wrong in the most convincing possible way: this server
ships unavailable fields as KEYS OF AN `unavailable` MAP, each carrying a `code`
and a `detail` explaining the absence. So a field that is explicitly declared
MISSING is textually identical to a field that is present as a measurement. A
substring search reports "on the wire" for both and inverts the answer for every
honestly-declared absence.

A key's meaning lives in the object that contains it. Parse the structure.

EXIT CODES
----------
  0  ran, and every classification agreed with the wire
  1  ran, and found a disagreement (or resolved nothing, which is not success)
  2  could not run -- no server reachable, or not inside a git worktree
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tree_context  # noqa: E402

PROVENANCE_PATH = "examples/serving-dashboard/telemetry-provenance.js"

# Endpoints the dashboard actually polls. A field is "served" if any of them
# sends it; a field is "declared unavailable" if any of them says so.
ENDPOINTS = (
    "/v1/status",
    "/v1/resources",
    "/v1/debug/kv",
    "/v1/debug/config",
    "/v1/models",
)

DEFAULT_PORTS = (8133, 8134, 8123, 8124, 8143, 8152)

# A run that compares nothing must not pass. This is the empty-input floor, and
# it is the check that has had to be reinvented in four separate guards on this
# branch. Anchored to the number of classified entries that carry a wire path.
MIN_COMPARISONS = 8


class CannotRun(Exception):
    """Raised when the check has no subject, so its silence would be vacuous."""


def load_provenance(repo: Path) -> dict:
    """Execute the table from COMMITTED BYTES, never from the working tree.

    `import()` reads the disk. Two agents on this branch executed the same module
    at the same sha from dirty trees and reported opposite results. Extracting
    the commit first removes the entire question.
    """
    try:
        source = subprocess.run(
            ["git", "show", f"HEAD:{PROVENANCE_PATH}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise CannotRun(
            f"could not read {PROVENANCE_PATH} from HEAD: {exc.stderr.strip()}"
        ) from exc

    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "provenance.mjs"
        module.write_text(source)
        script = (
            f"const m = await import({json.dumps(str(module))});"
            "const P = m.PROVENANCE ?? m.default;"
            "process.stdout.write(JSON.stringify(P));"
        )
        try:
            out = subprocess.run(
                ["node", "--input-type=module", "-e", script],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except FileNotFoundError as exc:
            raise CannotRun("node is not on PATH; cannot execute the table") from exc
        except subprocess.CalledProcessError as exc:
            raise CannotRun(
                f"the provenance module did not execute: {exc.stderr.strip()}"
            ) from exc
    return json.loads(out)


def fetch(origin: str) -> dict:
    """Return {endpoint: parsed json}. Unreachable endpoints are simply absent."""
    payloads = {}
    for endpoint in ENDPOINTS:
        try:
            with urllib.request.urlopen(f"{origin}{endpoint}", timeout=5) as response:
                payloads[endpoint] = json.load(response)
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return payloads


def find_origin(explicit: str | None) -> tuple[str, dict]:
    candidates = [explicit] if explicit else [f"http://127.0.0.1:{p}" for p in DEFAULT_PORTS]
    for origin in candidates:
        payloads = fetch(origin)
        if payloads:
            return origin, payloads
    raise CannotRun(
        "no server answered on "
        + ", ".join(candidates)
        + ".\n"
        "  This check compares the provenance table against a LIVE wire contract,\n"
        "  so with no server its result would be vacuous rather than clean.\n"
        "  Start a server (examples/serving-dashboard/run-demo.sh) or pass an\n"
        "  origin explicitly:  check_provenance_wire.py --origin http://127.0.0.1:8134"
    )


def wire_sets(payloads: dict) -> tuple[set[str], set[str]]:
    """Split the wire into (declared-unavailable, actually-served).

    The `unavailable` map is the server stating, in its own voice, which fields
    it is NOT sending and why. Those keys are not measurements and must never be
    counted as such -- that is the whole point of parsing rather than grepping.
    """
    declared_unavailable: set[str] = set()
    served: set[str] = set()

    def walk(node, prefix=""):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key == "unavailable" and isinstance(value, dict):
                declared_unavailable.update(value.keys())
                continue
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                served.add(path)
                walk(value, f"{path}.")
            elif isinstance(value, list):
                served.add(path)
            elif value is not None:
                served.add(path)

    for payload in payloads.values():
        walk(payload)
    return declared_unavailable, served


def classifications_of(entry: dict) -> set[str]:
    """An entry classifies once, or once per origin. Collect every claim it makes."""
    found = set()
    if isinstance(entry.get("classification"), str):
        found.add(entry["classification"])
    by_origin = entry.get("byOrigin")
    if isinstance(by_origin, dict):
        for origin_entry in by_origin.values():
            if isinstance(origin_entry, dict) and isinstance(
                origin_entry.get("classification"), str
            ):
                found.add(origin_entry["classification"])
    return found


def leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def check(provenance: dict, declared_unavailable: set[str], served: set[str]) -> tuple[list[str], int]:
    failures: list[str] = []
    compared = 0

    for key, entry in provenance.items():
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        classes = classifications_of(entry)
        if not classes:
            continue

        name = leaf(path)
        is_declared_absent = name in declared_unavailable
        is_served = (path in served or name in served) and not is_declared_absent

        if not (is_declared_absent or is_served):
            # The server neither sends it nor names it. Nothing to compare
            # against, so this entry is out of scope rather than wrong.
            continue

        compared += 1

        if "NOT_PLUMBED" in classes and is_served:
            failures.append(
                f"UNDERCLAIM - '{key}' is classified NOT_PLUMBED, but the server "
                f"serves '{name}' as a real value. The table denies a field that "
                f"is working; a document that understates our honesty is still wrong."
            )
        if "MEASURED" in classes and is_declared_absent:
            failures.append(
                f"OVERCLAIM - '{key}' is classified MEASURED, but the server "
                f"declares '{name}' in its `unavailable` map. The footer would "
                f"certify a number the server never sent."
            )

    return failures, compared


def run(origin_arg: str | None) -> int:
    repo = tree_context.repo_root()
    provenance = load_provenance(repo)
    origin, payloads = find_origin(origin_arg)
    declared_unavailable, served = wire_sets(payloads)

    if not declared_unavailable and not served:
        raise CannotRun(
            f"{origin} answered, but no endpoint yielded parseable fields. "
            "Refusing to report agreement over an empty wire."
        )

    failures, compared = check(provenance, declared_unavailable, served)

    print(f"origin {origin} - {len(payloads)}/{len(ENDPOINTS)} endpoints answered")
    print(
        f"wire: {len(served)} served fields, "
        f"{len(declared_unavailable)} declared unavailable"
    )
    print(f"provenance: {len(provenance)} entries, {compared} comparable against this wire")

    if compared < MIN_COMPARISONS:
        print(
            f"\nFAIL - only {compared} entries could be compared (floor {MIN_COMPARISONS}).",
            file=sys.stderr,
        )
        print(
            "  A guard that compares almost nothing reports success for the same\n"
            "  reason a broken one does. Either the wire contract moved or the\n"
            "  table's `path` fields no longer name wire fields.",
            file=sys.stderr,
        )
        return 1

    if failures:
        print(f"\nFAIL - {len(failures)} classification(s) disagree with the wire:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"\nOK - all {compared} comparable classifications agree with what "
        f"{origin} actually sends."
    )
    return 0


def self_test() -> int:
    """Prove the check discriminates, including a control that must NOT fire."""
    unavailable = {"tokens_per_second", "kv_usage"}
    served = {"batch_capacity", "queue_depth", "node_id"}

    cases = [
        (
            "control: honest table must pass",
            {
                "a": {"path": "batch_capacity", "classification": "MEASURED"},
                "b": {"path": "tokens_per_second", "classification": "NOT_PLUMBED"},
            },
            0,
        ),
        (
            "overclaim: MEASURED over a declared-unavailable field must fail",
            {"a": {"path": "tokens_per_second", "classification": "MEASURED"}},
            1,
        ),
        (
            "underclaim: NOT_PLUMBED over a served field must fail",
            {"a": {"path": "batch_capacity", "classification": "NOT_PLUMBED"}},
            1,
        ),
        (
            "control: a field the wire never mentions is out of scope, not a failure",
            {"a": {"path": "never_on_this_wire", "classification": "NOT_PLUMBED"}},
            0,
        ),
        (
            "byOrigin classifications are read, not skipped",
            {
                "a": {
                    "path": "tokens_per_second",
                    "byOrigin": {"dynamic": {"classification": "MEASURED"}},
                }
            },
            1,
        ),
    ]

    passed = 0
    for name, table, expected_failures in cases:
        failures, _ = check(table, unavailable, served)
        got = 1 if failures else 0
        status = "ok  " if got == expected_failures else "FAIL"
        if got == expected_failures:
            passed += 1
        print(f"  {status} {name}")
        if got != expected_failures:
            print(f"       expected failures={expected_failures}, got {failures}")

    print(f"\nself-test: {passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    origin = None
    if "--origin" in argv:
        origin = argv[argv.index("--origin") + 1]
    try:
        return run(origin)
    except (CannotRun, tree_context.NoWorktree) as exc:
        print(f"CANNOT RUN - {exc}", file=sys.stderr)
        return tree_context.CANNOT_RUN


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
