"""Which tree am I checking, and is it the one anyone ships?

Every check in this directory used to resolve the repository with
`git rev-parse --show-toplevel` in the CURRENT WORKING DIRECTORY. That is one
line and it is the single most expensive defect this project has had.

There are eight worktrees of this repository on this machine, at eight
different commits. A checker resolved from the CWD reports on whichever tree
the caller happened to be standing in, and its output is indistinguishable
either way: same format, same green, same exit code. Five false negatives in
one session came from one parked checkout, and every one of them looked like a
clean, correct, well-formed command. There was nothing wrong with the commands.

A COMMAND CANNOT TELL YOU IT IS POINTED AT THE WRONG UNIVERSE, AND NOTHING IN
ITS OUTPUT EVER WILL. So this module removes the choice rather than documenting
it: the tree is derived from THIS FILE'S OWN LOCATION on disk. A checker reads
the tree it lives in. Copy the script into a parked worktree and it honestly
checks the parked worktree; run the shipping tree's checker from anywhere at
all, including /tmp, and it still checks the shipping tree.

The banner exists for the other half of the problem. Results get pasted into
messages and outlive the shell they came from, so a result that does not carry
its own tree, branch and sha becomes an unfalsifiable claim the moment it is
quoted. Printing the provenance is cheaper than asking every agent to remember
to state it, and it cannot be forgotten.

On the branch check specifically: it is OPT-IN, and deliberately not a
hardcoded constant. A checker that asserts `branch == feat/genai-demo-dashboard`
starts failing on `main` the day the work merges -- a guard whose first act on
success is to break is a guard someone will delete in a hurry, and it takes the
real checks with it. Callers that want the assertion pass the branch in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Exit code for "this check could not run", kept distinct from 1, which every
# checker here uses for "this check ran and found a defect". A tool that cannot
# start and a tool that found a real problem must be separable by a caller --
# a CI job, a hook, or an agent -- without reading prose.
CANNOT_RUN = 2


class NoWorktree(Exception):
    """Raised when there is no git worktree to resolve tracked paths against."""


def repo_root() -> Path:
    """The worktree containing THIS FILE, never the caller's CWD.

    Fails with a NAMED diagnostic rather than a traceback when there is no
    worktree at all. That case is not hypothetical: the ratified way to review
    this branch is to extract a commit and measure the extract, and an archive
    extract is not a git repository. Every checker here needs `git ls-files` to
    know which paths are real, so refusing is correct -- but a raw
    CalledProcessError reads as "the tooling is broken" rather than "you are
    outside a repository", and it exits 1, which is byte-identical to a checker
    that ran and found a genuine defect. A crash and a finding must never print
    the same thing.
    """
    here = Path(__file__).resolve().parent
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=here,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise NoWorktree(
            f"{Path(__file__).name} is at {here}, which is not inside a git "
            f"worktree, so tracked-path resolution is impossible and every "
            f"result would be vacuous.\n"
            f"If this is an archive extract (`git archive`/`tar`), clone or "
            f"`git worktree add --detach <sha>` instead -- that preserves the "
            f"index these checks resolve against.\n"
            f"git said: {out.stderr.strip() or '(no stderr)'}"
        )
    return Path(out.stdout.strip())


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    return out.stdout.strip() if out.returncode == 0 else "?"


def tree_context(root: Path | None = None) -> dict:
    root = root or repo_root()
    porcelain = _git(root, "status", "--porcelain")
    return {
        "root": str(root),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git(root, "rev-parse", "--short", "HEAD"),
        "dirty": 0 if porcelain in ("", "?") else len(porcelain.splitlines()),
    }


def banner(ctx: dict | None = None) -> str:
    c = ctx or tree_context()
    # `dirty` is on the banner because a clean worktree at a stale HEAD is a
    # spotless measurement of the past, and a dirty one means the checker and
    # the committed state disagree about what was even read.
    return (
        f"tree {c['root']}\n"
        f"     branch {c['branch']} @ {c['sha']}, {c['dirty']} file(s) uncommitted"
    )


class WrongTree(Exception):
    pass


def require_branch(expected: str, ctx: dict | None = None) -> None:
    """Fail loudly when checking a tree nobody ships. Opt-in, never implicit."""
    c = ctx or tree_context()
    if c["branch"] != expected:
        raise WrongTree(
            f"this check is reading {c['root']}, which is on branch "
            f"'{c['branch']}' @ {c['sha']}, but was told to verify "
            f"'{expected}'. Refusing to report on a tree nobody ships: a "
            f"green result from the wrong worktree is byte-identical to a "
            f"green result from the right one."
        )
