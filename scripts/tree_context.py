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


# ---------------------------------------------------------------------------
# Am I measuring the thing we ship?
# ---------------------------------------------------------------------------
#
# Every checker in this directory answers a question about the repository, and
# each one had to decide for itself where "the repository" lives. About half
# got it wrong, and all of them got it wrong in the SAME direction and for the
# same reason: they ENUMERATE from the committed tree (`git ls-files`) and then
# READ FROM THE WORKING DIRECTORY. That pairing looks careful. It is the worst
# of both: the file list is what we ship, and the bytes are whatever happens to
# be on one agent's desk at the moment the checker ran.
#
# In a shared worktree with fourteen agents editing concurrently, an
# uncommitted edit is a rumour. A checker that reads it is scoring a draft that
# may never exist again, and PUBLISHING THE RESULT AS A FACT ABOUT THE BRANCH.
# This is not hypothetical here: a citation check went green over a document
# whose cited source file was dirty at the time, and a freeze was declared on
# that green.
#
# The fix did not travel the first time because it lived inside one checker as
# a private `git show` call. A FIX THAT IS NOT IMPORTABLE IS A FIX THAT APPLIES
# EXACTLY ONCE. So it lives here now.
#
# WHY NOT SIMPLY ALWAYS READ HEAD. Because that breaks a legitimate and common
# workflow -- adding a symbol and citing it in the same commit -- by reporting
# a defect that does not exist, and a guard that goes red on correct work gets
# switched off. So the helper does not silently pick a side. It reads what the
# caller asks for AND MAKES THE DISAGREEMENT LOUD, because the only genuinely
# dangerous state is the one where the two trees differ and the report does not
# say so.


class NotInRef(Exception):
    """A path exists on disk but not in the ref being measured."""


def shipped_text(root: Path, relpath: str, ref: str = "HEAD") -> str:
    """The bytes of `relpath` AS COMMITTED at `ref` -- never the desk copy.

    Raises rather than returning "" when the path is absent from the ref. That
    distinction is the whole point of the function: an empty string is a
    perfectly valid file body, so returning it for "this file does not exist
    here" hands the caller a silent wrong answer in the one direction that
    matters. Every checker in this directory searches text for something it
    expects to find, and searching "" is indistinguishable from searching a
    file that genuinely lost the thing you were looking for.
    """
    out = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if out.returncode != 0:
        raise NotInRef(
            f"{relpath!r} is not present in {ref}. If it is a new file, it is "
            f"not part of what we ship yet; if it was deleted, any result "
            f"about it is stale.\ngit said: {out.stderr.strip() or '(none)'}"
        )
    return out.stdout


def divergent_paths(root: Path, paths, ref: str = "HEAD") -> dict:
    """Which of `paths` disagree between the desk and `ref`, and how.

    Returns {"modified": [...], "untracked": [...], "deleted": [...]}.

    `git status --porcelain` is asked about the paths THE CALLER ACTUALLY READ,
    not about the tree at large. A checker that reads six files does not care
    that a seventh is dirty, and a global "9 files uncommitted" banner is so
    routinely non-zero on this branch that it has stopped carrying information.
    Scoping the question to the inputs is what turns it back into a signal.
    """
    paths = [str(p) for p in paths]
    result = {"modified": [], "untracked": [], "deleted": []}
    if not paths:
        return result
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return result
    for line in out.stdout.splitlines():
        code, _, name = line[:2], line[2], line[3:].strip().strip('"')
        if code == "??":
            result["untracked"].append(name)
        elif "D" in code:
            result["deleted"].append(name)
        else:
            result["modified"].append(name)
    return result


def divergence_report(root: Path, paths, ref: str = "HEAD") -> list:
    """Human-readable disclosure lines. EMPTY MEANS AGREEMENT, not 'unchecked'.

    Callers must print these unconditionally alongside their verdict. A result
    that quietly measured uncommitted bytes is not wrong -- it is UNFALSIFIABLE,
    which is worse, because nobody can tell later which tree it described.
    """
    d = divergent_paths(root, paths, ref)
    lines = []
    for name in sorted(d["modified"]):
        lines.append(
            f"WORKTREE_DIVERGENCE {name}: read from the working tree, which "
            f"differs from {ref}. This result describes uncommitted bytes."
        )
    for name in sorted(d["untracked"]):
        lines.append(
            f"WORKTREE_DIVERGENCE {name}: untracked. Absent from {ref} "
            f"entirely -- it is on one desk and is not part of what we ship."
        )
    for name in sorted(d["deleted"]):
        lines.append(
            f"WORKTREE_DIVERGENCE {name}: deleted on disk but present in "
            f"{ref}. Any result about it describes a file that is going away."
        )
    return lines
