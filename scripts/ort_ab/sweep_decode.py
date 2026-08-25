#!/usr/bin/env python3
"""Sweep decode-shape matmul cells across thread counts, ours vs ORT.

`bench_generic` already alternates our EP and ORT inside one process, so each
line is a paired measurement. What this adds is the *thread-count sweep* and an
aggregation that survives a contended host: it reports the median of the
per-run p50, the median of the per-run p90 (the statistic
`docs/performance/CPU_MATMUL_ASSIGNMENT.md` tabulates) **and** the min of the
per-run min, because on a shared machine the min is the only statistic that is
not partly a measure of the other tenants.

Absolute milliseconds are printed, not just the ratio -- the whole question
these rows raise is whether we get slower with more threads or whether ORT
simply scales while we stay flat, and a ratio cannot tell those apart.

The min-of-min above was written before the host lock existed, and it is a way
to *survive* contention rather than to exclude it: it is blind to an SMT
sibling and to steady external load, which is exactly what `73c76458c` says
about per-run efficiency and A/A nulls. So this sweep now refuses to run
without a declaration held by an ancestor -- a thread sweep is the most
saturating thing we run, and it is the least defensible thing to run beside
somebody else's measurement. `--unlocked` exists for smoke tests and stamps
every row.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

import hostlock_gate

LINE = re.compile(
    r"native=(?P<native>[\d.]+) ms .*?ort=(?P<ort>[\d.]+) ms .*?"
    r"native/ort=(?P<ratio>[\d.]+) "
    r"native_p90=(?P<native_p90>[\d.]+) ort_p90=(?P<ort_p90>[\d.]+) "
    r"native_min=(?P<native_min>[\d.]+) ort_min=(?P<ort_min>[\d.]+)"
)

# `bench_generic` already reports what the width request actually became --
# `native_width_as_requested` is `no` when the pool or the task runtime handed
# back fewer lanes than were asked for. This sweep threw those fields away and
# labelled every row with the number it *asked* for, which is the defect that
# ate four published decode rows: at t=1 the dispatcher takes a serial
# short-circuit, so the t=1 column measures a different code path from every
# other column in the same table. A thread sweep whose leftmost column is a
# different route is not a scaling curve.
WIDTH = re.compile(
    r"native_pool_width=(?P<pool_width>\S+) native_path=(?P<path>\S+) "
    r"native_task_width=(?P<task_width>\S+) native_cpus=(?P<cpus>\S+) "
    r"native_width_as_requested=(?P<as_requested>\S+)"
)


def run_one(
    binary: Path,
    model: Path,
    threads: int,
    runs: int,
    warmups: int,
    timeout: float | None = None,
):
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--native-threads",
        str(threads),
        "--ort-intra-threads",
        str(threads),
        "--runs",
        str(runs),
        "--warmups",
        str(warmups),
    ]
    # A wedged bench would otherwise hang the sweep *while the wrapper holds
    # the shared host lock* -- squatting on the box is the one outcome the
    # whole lock discipline exists to prevent, and a hung child is the easiest
    # way to do it by accident.
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=timeout
    ).stdout
    for line in out.splitlines():
        m = LINE.search(line)
        if m:
            cell = {k: float(v) for k, v in m.groupdict().items()}
            cell["width"] = width_report(line)
            return cell
    raise RuntimeError(f"no result line for {model} t={threads}")


def width_report(line: str) -> dict[str, str]:
    """The realized-width fields, or a marker saying the binary did not emit them.

    `absent` is not the same as `no`, and neither is the same as a missing
    column: an older `bench_generic` that never reported width would otherwise
    read as a satisfied request. It is a distinct value so the sweep can say
    "this binary cannot tell me" rather than assuming either answer.
    """
    m = WIDTH.search(line)
    if not m:
        return {
            "as_requested": "absent",
            "pool_width": "absent",
            "path": "absent",
            "task_width": "absent",
            "cpus": "absent",
        }
    return m.groupdict()


def width_verdict(
    requested: int, cells: list[dict[str, str]], require: bool = False
) -> tuple[str, str | None, bool]:
    """Whether this cell's rows describe the route their `t` column names.

    Returns `(label, caveat, fatal)`. The label goes on the row so a table
    pasted into a doc carries it; the caveat says what the label means; and
    `fatal` says whether the sweep may still leave with exit 0.

    **A reduced width is not a failure here, and that distinction is the whole
    design.** Asking for 16 lanes on a box whose SMT cap or cpuset affords 8
    is the engine behaving correctly on a shared machine, which is the only
    kind we have (#1802): failing the sweep for it would make the check
    satisfiable only on a large idle host, i.e. exactly the exclusive-host
    assumption we are not allowed to bake in. That cell is still on the pooled
    route and still comparable; it just carries a narrower claim, so it is
    labelled `capped` and says so.

    What *is* fatal is a **route** difference, because it is categorical and
    host-independent:

    - `not-requested` -- the bench says no width was requested at all, though
      we passed `--native-threads`. The knob never reached the engine.
    - `varied` -- the trials inside one cell did not agree, so their median is
      not of one thing.

    The cross-cell route check in `main` is the third, and it is the one that
    would have caught the original defect: `--native-threads 1` takes the
    dispatcher's serial short-circuit rather than a one-worker pool, so the
    t=1 row is a different `native_path` from every other row in the table.

    `require=True` (`--require-width`) hardens `capped` and `absent` into
    failures for the caller who genuinely needs exact lanes. It is opt-in
    because it is a demand on the host, not on the engine.
    """
    answers = {c.get("as_requested", "absent") for c in cells}
    paths = sorted({c.get("path", "?") for c in cells})
    pool = sorted({c.get("pool_width", "?") for c in cells})
    task = sorted({c.get("task_width", "?") for c in cells})

    if answers == {"yes"} and len(paths) == 1:
        return "yes", None, False
    if len(answers) > 1 or len(paths) > 1:
        # Differing *paths* count even when every trial says `yes`: two trials
        # can each get the width they asked for and still be on different
        # routes, and a median across those is not a measurement of either.
        return "varied", (
            f"t={requested}: the trials in this cell were not all on the same "
            f"route (paths {paths}, pool widths {pool}, realized "
            f"{sorted(answers)}), so their median is not of one thing."
        ), True
    if answers == {"n/a"}:
        if requested == 0:
            # `--native-threads 0` is the documented way to opt out of the
            # bounded pool, so `n/a` is the honest answer to a question we
            # chose not to ask. Calling that "the knob did not reach the
            # engine" would be a false statement about a deliberate input.
            return "opted-out", None, False
        return "not-requested", (
            f"t={requested}: the bench reports that no width was requested, "
            "though this sweep passed --native-threads. The knob did not "
            "reach the engine, so every column in this table is one route."
        ), True
    if "absent" in answers:
        return "absent", (
            f"t={requested}: this bench binary does not report realized "
            "width, so the column label is unverified -- which is not the "
            "same as wrong. Rebuild from a revision that emits "
            "native_width_as_requested to check it."
        ), require
    cpus = sorted({c.get("cpus", "?") for c in cells})
    # Both realized widths, never one: `as_requested=no` means the pool width
    # OR the task width missed the request (bench_generic.rs:707-711), so
    # naming a single number would sometimes print "got 16, not 16".
    return "capped", (
        f"t={requested}: the request was not realized -- route {paths[0]}, "
        f"pool width {pool[0]}, task width {task[0]}, on a host reporting "
        f"{cpus[0]} cpus. An SMT or cpuset cap is the engine's own policy "
        "doing its job, not a defect -- but this row's claim is about the "
        f"widths it got, not about {requested}."
    ), require


def exit_code(window_code: int, structural: bool) -> int:
    """Which single finding a streamed sweep leaves with.

    Only one number gets out, so the order is a claim about which finding the
    caller most needs, and it is worth stating rather than leaving to whatever
    check happened to run last:

    - **4 (custody changed) wins over everything.** Every row is discarded, so
      a finding about columns nobody will quote would only bury the one that
      matters.
    - **6 (route) beats 5 (unreadable end of window).** 6 is a definite
      structural fact about what ran; 5 is "I could not tell". Reporting the
      maybe in place of the certainty understates what is known.
    """
    if window_code == 4:
        return 4
    return 6 if structural else window_code


def route_verdict(observed: dict[int, set[str]]) -> str | None:
    """The check that would have caught the defect this file exists to prevent.

    `observed` maps requested width to every `native_path` its cells reported
    (a set, because the same width is swept for each model).
    A sweep is a scaling curve only if every column is the same route; when
    `t=1` is `flat` because the dispatcher short-circuits and every other
    column is `pool`, the leftmost point is a different program, and the
    curve through it describes nothing.

    Categorical, and independent of load: it compares the routes with each
    other, not against any expectation of the host.
    """
    by_route: dict[str, list[int]] = {}
    for width, paths in sorted(observed.items()):
        for path in sorted(paths):
            if path in ("?", "absent"):
                continue
            by_route.setdefault(path, []).append(width)
    if len(by_route) < 2:
        return None
    detail = "; ".join(
        f"{path}: t={','.join(str(w) for w in widths)}"
        for path, widths in sorted(by_route.items())
    )
    return (
        f"the columns are not all the same route ({detail}). A sweep across "
        "code paths is not a scaling curve -- the points do not lie on one "
        "function of thread count."
    )


def end_of_window_verdict(start: str, end: str) -> tuple[int, str | None]:
    """What to say once the rows are already on the screen.

    `ab.py` buffers, so it can stamp the end-of-window label onto every row.
    This driver streams, so it cannot relabel what has been printed -- which
    makes the exit code and one stderr line the only honest places left to
    put the finding.

    The three outcomes are kept distinct because they mean different things
    to the person reading them. A handoff (`changed`) means the thread counts
    above and below the change were compared across it: not half-good data,
    discard it. A failed end-read (`unverified-end`) means we do not know --
    the rows may be perfectly good, and telling someone to discard them would
    assert a specific false fact about data that is probably fine. Collapsing
    the second into the first is the exact conflation `window_label` and its
    tests exist to prevent.
    """
    if end == "changed":
        return 4, (
            f"host_lock=changed: the declaration covering this sweep did not "
            f"hold for the whole of it (started {start}). Every row above "
            "spans the change -- discard them."
        )
    if end == "unverified-end":
        return 5, (
            f"host_lock=unverified-end: the lock could not be re-read when "
            f"the sweep finished, so the {start} label on every row above is "
            "unverified at the far end. The rows may be sound; nothing here "
            "establishes that they are."
        )
    return 0, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=Path("target/release/bench_generic"))
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument(
        "--cell-timeout",
        type=float,
        default=3600.0,
        help="seconds one bench invocation may take before the sweep gives up "
        "(it holds the host lock while it waits); 0 disables",
    )
    ap.add_argument(
        "--require-width",
        action="store_true",
        help="treat a capped or unreportable width as a failure. Opt-in: it is "
        "a demand on the host, not on the engine, and a shared or "
        "cpuset-confined box can legitimately not meet it",
    )
    ap.add_argument(
        "--unlocked",
        action="store_true",
        help="run without a host-lock declaration and stamp every row (smoke tests only)",
    )
    args = ap.parse_args()

    # Before the first child, not after: refusing later has already put load on
    # a host somebody else declared.
    lock_label, prov = hostlock_gate.require(
        "python3 scripts/ort_ab/sweep_decode.py <your args>", unlocked=args.unlocked
    )
    columns = hostlock_gate.lock_columns(lock_label, prov)
    print(
        " ".join(f"{k}={v}" for k, v in columns.items()),
        flush=True,
    )

    print(
        f"{'model':26s} {'t':>3s} {'native_p50':>10s} {'ort_p50':>8s} "
        f"{'ratio_p50':>9s} {'ratio_p90':>9s} {'native_min':>10s} {'ort_min':>8s} "
        f"{'ratio_min':>9s} {'width_ok':>8s} {'host_lock':>14s}"
    )
    width_complaints: list[str] = []
    width_fatal = False
    routes: dict[int, set[str]] = {}
    for model in args.models:
        for threads in args.threads:
            trials = []
            for _ in range(args.trials):
                trials.append(
                    run_one(
                        args.binary,
                        model,
                        threads,
                        args.runs,
                        args.warmups,
                        timeout=args.cell_timeout or None,
                    )
                )
            native_p50 = statistics.median(t["native"] for t in trials)
            ort_p50 = statistics.median(t["ort"] for t in trials)
            native_p90 = statistics.median(t["native_p90"] for t in trials)
            ort_p90 = statistics.median(t["ort_p90"] for t in trials)
            native_min = min(t["native_min"] for t in trials)
            ort_min = min(t["ort_min"] for t in trials)
            width_ok, caveat, fatal = width_verdict(
                threads, [t["width"] for t in trials], require=args.require_width
            )
            if caveat:
                width_complaints.append(caveat)
            width_fatal = width_fatal or fatal
            routes.setdefault(threads, set()).update(
                t["width"].get("path", "?") for t in trials
            )
            print(
                f"{model.stem:26s} {threads:3d} {native_p50:10.3f} {ort_p50:8.3f} "
                f"{native_p50 / ort_p50:9.3f} {native_p90 / ort_p90:9.3f} "
                f"{native_min:10.3f} {ort_min:8.3f} "
                f"{native_min / ort_min:9.3f} {width_ok:>8s} {lock_label:>14s}",
                flush=True,
            )

    end_label = hostlock_gate.window_label(
        lock_label, prov, hostlock_gate.read_provenance()
    )
    code, complaint = end_of_window_verdict(lock_label, end_label)
    if complaint:
        print(complaint, file=sys.stderr)

    # Reported after the table rather than raised mid-sweep: a cell whose
    # width was capped is still worth seeing next to the ones that were not --
    # that comparison is often what identifies the mechanism.
    for line in width_complaints:
        print(line, file=sys.stderr)

    route_complaint = route_verdict(routes)
    if route_complaint:
        print(route_complaint, file=sys.stderr)

    return exit_code(code, width_fatal or route_complaint is not None)


if __name__ == "__main__":
    sys.exit(main())
