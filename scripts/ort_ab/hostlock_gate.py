#!/usr/bin/env python3
"""The host-lock admission gate every benchmark driver has to pass.

`73c76458c` made the lock mandatory for saturating runs, with the outer
harness holding it across every arm. That is a property of the *driver*, and
until this module existed the only thing enforcing it was remembering to type
the wrapper -- which is the same kind of guarantee as checking `ps` for a
running bench binary, and it fails the same way: #1803 was filed because a
peer looked at a quiet host and started a sweep in the gap between two arms
of somebody's interleaved A/B. Nobody was careless. The check could not give
the answer it was being asked for.

So the check here is ancestry, not liveness. A lock held by a benchmark
*child* is released between arms and certifies each arm while protecting none
of the comparison; a lock held by a *peer* is a reason to stop rather than to
start. Only a `HELD` declaration anchored to this process or one of its
ancestors admits.

Extracted from `ab.py`, which is where it was first written (#2032) and
where its behaviour is pinned by `test_ab_lock.py`.

Usage, in the driver's `main()` before anything is launched:

    label, prov = hostlock_gate.require(
        "python3 scripts/ort_ab/sweep_decode.py <your args>",
        unlocked=args.unlocked,
    )

and, when the matrix is done, `window_label(label, prov, read_provenance())`
to catch a lock that changed hands mid-run, then `lock_columns` to stamp the
result onto every emitted row. A driver that does not emit rows should still
print the label: a console line is worth less than a column, but it is worth
more than nothing.

Usable from outside this directory, and deliberately so: the #2043 ledger
records nineteen ungated harnesses in `crates/onnx-runtime-ep-cpu/benches/`,
and asking their owners to write a lock client each is asking for nineteen
subtly different ones. Nothing here reads the working directory --
`hostlock.sh` is resolved from this file -- so a harness in another root
needs a path insert and the same two calls:

    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve()
    _root = next(p for p in _here.parents if (p / "scripts" / "ort_ab").is_dir())
    sys.path.insert(0, str(_root / "scripts" / "ort_ab"))
    import hostlock_gate

    label, prov = hostlock_gate.require(
        "python3 crates/onnx-runtime-ep-cpu/benches/<this file> <args>",
        unlocked=args.unlocked,
    )

The ascent is deliberate where a `parents[3]` would read more simply: that
constant is correct only for a file sitting directly in the benches root, and
a harness one directory deeper gets a path insert pointing at `crates/` and a
bare `ImportError` at the top of its `main()`. A recipe that is copied is a
recipe that will be copied somewhere else.

The gate *checks* custody; it never takes the lock, because a lock taken by
the driver is released when the driver exits and certifies nothing about a
matrix run as several processes. `scripts/hostlock.sh run -- <driver>` is
what holds it, and `remedy()` prints that line with the caller's own command
in it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HOSTLOCK = Path(__file__).resolve().parents[1] / "hostlock.sh"


def parse_provenance(text: str) -> dict[str, str]:
    """`hostlock.sh provenance --oneline` into a dict.

    Values cannot contain spaces: the script sanitises owner and reason for
    exactly this reason, so splitting on whitespace is the format's contract
    rather than an assumption about it.
    """
    fields = {}
    for token in text.split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    return fields


def read_provenance(runner=subprocess.run) -> dict[str, str]:
    """Asks the lock what it is doing. Empty on any failure.

    An empty reading is fail-closed here: `lock_verdict` refuses anything it
    cannot read as a live declaration held by this harness, so a missing or
    broken `hostlock.sh` stops the run instead of silently ungating it.
    """
    try:
        out = runner(
            ["bash", str(HOSTLOCK), "provenance", "--oneline"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return {}
    return parse_provenance(out.stdout)


def parent_of(pid: int) -> int | None:
    """`/proc/<pid>/stat` field 4.

    The comm field is parenthesised and may itself contain spaces and
    parentheses, so the split is anchored on the LAST `)` -- a naive
    `split()[3]` reads the wrong column for a process whose name has a space
    in it, and reads a plausible number rather than failing.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    _, _, rest = stat.rpartition(")")
    parts = rest.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def ancestry(pid: int, parent=parent_of, limit: int = 64) -> set[int]:
    """This process and every ancestor of it, up to init.

    `limit` and the seen-set are not paranoia: pid 1's parent is 0, a
    namespaced or reparented process can report a parent that is already in
    the chain, and a walk that trusted the chain to terminate would hang the
    harness before it ran anything.
    """
    chain = {pid}
    current = pid
    for _ in range(limit):
        nxt = parent(current)
        if nxt is None or nxt <= 0 or nxt in chain:
            break
        chain.add(nxt)
        current = nxt
    return chain


def lock_verdict(prov: dict[str, str], chain: set[int]) -> tuple[str, str | None]:
    """The `host_lock=` label for this run, and why it may not proceed.

    Returns `(label, None)` when the run is covered by a declaration held by
    this harness or one of its ancestors, and `(label, reason)` otherwise.

    The ancestry test is the point, and it is stronger than "is the lock
    held": a lock held by a *child* -- one `hostlock.sh run` per benchmark
    invocation -- is released between arms, so it certifies each arm and
    protects none of the comparison. A lock held by a *peer* is a reason to
    stop rather than to start.

    Only the pid *number* is compared, and that is safe **only** because it
    sits behind `state == "HELD"`: `hostlock.sh` reports HELD only for an
    anchor whose pid **and** `/proc` start time both still match, so a
    recycled pid reaches here as STALE and is refused. This box cycles ~1.5M
    pids in four days, so admitting on `held_pid in chain` without the state
    check -- or accepting a state the script does not start-time verify --
    would be a genuine false admit. The invariant is recorded because it is
    invisible in the expression that depends on it.
    """
    state = prov.get("hostlock_state", "")
    owner = prov.get("held_by", "none")
    try:
        anchor = int(prov.get("held_pid", "none"))
    except ValueError:
        anchor = 0

    if not prov:
        return "unknown", "the host lock could not be read at all"
    if state == "HELD" and anchor in chain:
        return f"mine:{owner}", None
    if state == "HELD":
        return (
            f"foreign:{owner}",
            f"{owner} (pid {anchor}) holds this host, and that declaration is "
            "not an ancestor of this harness",
        )
    if state == "EXPIRED":
        return (
            f"expired:{owner}",
            "the declaration covering this host has expired, so a peer may "
            "take the box mid-matrix. Re-acquire before measuring",
        )
    if state == "STALE":
        return (
            f"stale:{owner}",
            f"the lock is held by a dead anchor ({owner}, pid {anchor}). "
            "Reaping it does not stop whatever load it was covering, so "
            "check the host before taking it",
        )
    if state == "UNUSABLE":
        return (
            "unusable",
            "this host cannot take the lock at all (see `hostlock.sh status`). "
            "Fix `lock_dir=` rather than measuring without one",
        )
    if state == "FREE":
        return "free", "no declaration covers this run"
    return state.lower() or "unknown", f"the lock reports {state or 'nothing'}"


def window_label(label: str, before: dict[str, str], after: dict[str, str]) -> str:
    """`changed` when custody moved during the run, else `label` unchanged.

    A run that changed hands halfway through was protected for neither half,
    and a label naming whoever happened to hold the lock at one end describes
    the other end as something it was not. Both the owner and the anchor pid
    are compared: the same agent re-acquiring under a new anchor is still a
    gap in which the box was free, and on a host cycling ~1.5M pids in four
    days the pid alone can repeat.
    """
    if not after:
        # The second read failed. That is not evidence of a handoff, and
        # saying `changed` would assert a specific false fact -- custody
        # moved -- about data that may be perfectly good. It is equally not
        # evidence the declaration held, so the row says which it is.
        return "unverified-end"
    moved = (after.get("held_by"), after.get("held_pid")) != (
        before.get("held_by"),
        before.get("held_pid"),
    )
    return "changed" if moved else label


def lock_columns(label: str, prov: dict[str, str]) -> dict[str, str]:
    """The lock fields stamped onto every row.

    A dict built in one place, so the mapping from provenance key to column
    is testable. Built inline it was not: two of these columns could be
    swapped, or read from the wrong provenance key, with every cell still
    green -- the row would carry a number under a name that did not describe
    it, which is worse than carrying nothing.

    `contended` is deliberately absent. `hostlock.sh` only computes it when
    given `--expect-runnable`, and there is no honest threshold to pass here:
    this host is shared by design (#1802), so "more runnable than expected"
    has no fixed value. A column that reads `unknown` on every real run is an
    invitation to treat its absence as reassurance.
    """
    return {
        "host_lock": label,
        "lock_owner": prov.get("held_by", "none"),
        "lock_anchor_pid": prov.get("held_pid", "none"),
        # An instantaneous sample, named so it cannot be read as a property of
        # the window. It is the runnable count at the moment the matrix
        # started and says nothing about what happened afterwards.
        "runnable_at_start": prov.get("runnable", "unknown"),
    }


def remedy(command: str) -> str:
    """The message someone reads at the moment they are stopped.

    Long on purpose, and it names the caller's own command: a remedy that
    says "wrap it in the lock" without showing the exact line gets pasted
    wrong, and the most common wrong paste -- wrapping the benchmark child
    instead of the driver -- is the defect this gate exists to prevent.
    """
    return f"""
Wrap the WHOLE matrix -- every arm, including any null control -- in the lock:

    scripts/hostlock.sh run --owner <you> --reason "<what this measures>" -- \\
        {command}

Wrapping each benchmark child instead leaves the host looking idle in the gap
between two arms, which is how one agent started a sweep in the middle of
another's interleaved A/B. The holder must be the process that spans the arms.

`--unlocked` runs anyway and stamps every row so the numbers cannot later be
mistaken for protected ones. It is for smoke tests, not for anything you
intend to publish."""


def require(command: str, unlocked: bool = False) -> tuple[str, dict[str, str]]:
    """Admit this run or exit 3, before any arm is launched.

    Returns `(label, provenance)` on admission. The label is what every row
    must carry; hold on to the provenance so the end-of-run reading can be
    compared against it.

    Exit 3 rather than 1: a refusal is not a benchmark failure, and a caller
    scripting several matrices needs to tell "the host was not mine" apart
    from "the binary crashed".

    `unlocked` does not extend to a lock somebody else declared. The escape
    hatch exists so an unprotected run cannot later be quoted as a protected
    one -- an argument entirely about our own labels, which says nothing
    about a box a peer has taken and where the damage lands on their
    measurement instead.
    """
    prov = read_provenance()
    label, refusal = lock_verdict(prov, ancestry(os.getpid()))
    if refusal is None:
        return label, prov
    if unlocked and not label.startswith("foreign:"):
        label = f"unlocked:{label}"
        print(
            f"WARNING: running unlocked ({refusal}). Every row is stamped "
            f"host_lock={label} and none of them is publishable.",
            file=sys.stderr,
        )
        return label, prov
    print(f"refusing to measure: {refusal}.", file=sys.stderr)
    if label.startswith("foreign:"):
        print(
            "--unlocked does not override a peer's declaration.",
            file=sys.stderr,
        )
    print(remedy(command), file=sys.stderr)
    raise SystemExit(3)
