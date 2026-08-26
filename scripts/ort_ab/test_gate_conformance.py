#!/usr/bin/env python3
"""Every benchmark driver in this directory takes the host lock, or says why not.

The lock is mandatory for saturating runs, but "mandatory" has so far meant
that somebody remembered. The audit in #2043 found exactly one driver of
fourteen calling the gate, and nothing in the tree would have told us: an
ungated harness looks identical to a gated one until it is running next to
somebody else's matrix.

So this file makes the classification explicit and checks it. Every `.py` in
`scripts/ort_ab/` must be declared as a driver, a generator, a library or a
test. A driver must call `hostlock_gate.require`. A file declared as anything
else must not look like a driver -- if it imports the runtime, opens an
inference session, or names a binary under `target/`, the declaration is
contradicted by the file itself and the suite fails.

The point is not that this catches today's drivers; #2043 already listed
those. It is that the *next* harness cannot ship ungated by accident, because
an unclassified file is a failure and a misclassified one is a failure with a
reason attached.

Two roots are covered. `scripts/ort_ab/` is read by declaration: every `.py`
must be listed as a driver, a generator, a library or a test, and a driver
must call `hostlock_gate.require`.

The second root, `crates/onnx-runtime-ep-cpu/benches/`, is read by behaviour
instead of by declaration: it holds twenty-six analysis and harness scripts
that are not mine, and a per-file role list there would be a claim about
somebody else's lane that I would then have to keep true. So a file that
starts a process or opens a session must hold the lock or carry a recorded
gap, and a file that only reads JSON needs no entry at all. Recorded gaps are
checked in both directions: a gap that has since been gated **fails as
stale**, so closing one is a one-line ledger edit and the record cannot drift
into fiction.

That root is also read in shell. It was not, for as long as this file has
existed, because the enumeration globbed `*.py` -- and the one shell harness
there claimed the lock in its own header comment while calling nothing. The
shell rule is categorical rather than behavioural (every `*.sh` gates or is
recorded), for the reason written at `EP_SHELL_LEDGER`. The thirteen Rust
`[[bench]]` targets in the same directory are still uncovered, deliberately
and with the reason written down in the same place (#2129), because what must
hold the lock for those is the `cargo bench` invocation rather than the
source. Their *names* are pinned, so a fourteenth cannot arrive silently
while none of the thirteen gates.

Both lock idioms in the tree count as held: `hostlock_gate.require` in
`scripts/ort_ab/`, and the `HostLock` context manager in `acc0_gap_matrix.py`
that shells out to `scripts/hostlock.sh`. A checker that knew only its
author's idiom would report three genuinely gated harnesses as ungated, which
is how a conformance check gets deleted rather than obeyed.

**What it does not catch**, written down so nobody reads more into a green
run than is there:

- A driver **misdeclared as a generator** whose binary path is built at
  runtime (`os.path.join("target", "release", ...)`, an f-string, an env
  var). The contradiction check reads literals, so it would not object --
  `ab.py` itself has no `target/` literal. Declaring a driver a generator is
  a false statement in a reviewed file, which is the layer that catches it.
- Whether a held lock is *the right* lock, or held for the whole matrix. The
  loop check below is one property of custody (acquisition is not inside the
  arm loop), not the whole of it; the run-time answer is the `host_lock`
  column on every emitted row.
- A spawn behind deliberate indirection: `getattr(subprocess, "run")(cmd)`,
  a dict of callables, `importlib`. Every check here reads names in the tree,
  so a name assembled at runtime is invisible. This is the fail-open edge
  that cannot be closed by reading source, and it is why the ledger is a
  reviewed file rather than only a program.
- A file that saturates the box **in process**, without starting anything.
  `cpu_work_probe.py` is the honest example: it burns a fixed CPU-second in
  Python to measure delivered work. It is single-threaded and bounded, so it
  is a probe rather than a matrix, but a twenty-thread version of it would be
  invisible here -- as would a `threading.Thread` pool around an
  `InferenceSession` that did not import the runtime under a name this file
  knows. Detecting "starts something" is a proxy for "occupies the host", and
  this is the gap between them.

Run: `python3 scripts/ort_ab/test_gate_conformance.py`
"""

from __future__ import annotations

import ast
import contextlib
import functools
import re
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

ORT_AB = Path(__file__).resolve().parent

# Declared roles. A file in more than one list, or in none, is a failure.
#
# `known-gap:` is deliberately a *driver* declaration rather than an exemption:
# the file is a driver, it does not take the lock, and the string records who
# is expected to fix it. A gap that has to be written down is a gap somebody
# can act on; a gap that is merely absent is one nobody can see.
DRIVERS = {
    "ab.py": "gated",
    "sweep_decode.py": "gated",
    "ort_cuda_decode_bench.py": "known-gap:#2043 - GPU lane, owner to gate it",
}

LIBRARIES = {
    "hostlock_gate.py": "the gate itself",
}

# A non-driver may legitimately touch the runtime -- a generator that checks
# its fixture actually loads, say. `loads-runtime:` declares that, the same
# way `known-gap:` declares an ungated driver: written down, not enforced out
# of existence. Any other reason string does not suppress the finding.
RUNTIME_OK = "loads-runtime:"

TESTS = {
    "test_ab_lock.py": "admission cells for the gate",
    "test_gate_conformance.py": "this file",
}

# Fixture generators: they write .onnx files and measure nothing. `gen_gqa.py`
# was miscounted as a harness in the first pass of #2043 for having `bench` in
# a docstring, which is why the check below reads what a file *does* rather
# than what it is called.
GENERATORS = {
    name: "writes model fixtures, measures nothing"
    for name in (
        "gen_activations.py",
        "gen_decode.py",
        "gen_f16_gemv.py",
        "gen_f16_nt.py",
        "gen_gemm.py",
        "gen_gqa.py",
        "gen_grid.py",
        "gen_l3sweep.py",
        "gen_mha.py",
        "gen_moe.py",
        "gen_qlinear.py",
        "gen_sdpa_region.py",
        "gen_transforms.py",
    )
}

# What a driver looks like from the outside, whatever it is called: it opens
# an inference session, imports the runtime that measures one, or starts a
# binary out of `target/`.
#
# Read from the parsed tree rather than the text, because the first cut of the
# #2043 audit called `gen_gqa.py` a harness for saying "bench" in a docstring.
# A classification that can be tripped by prose is not one anyone will keep.
RUNTIME_MODULES = ("onnxruntime",)
SESSION_CALLS = ("InferenceSession",)
BUILT_BINARY = re.compile(r"target/(?:release|debug)/")


def takes_the_lock(source: str) -> bool:
    """A call to the gate, found in the parsed tree.

    Both import styles reach the same gate: `hostlock_gate.require(...)` and
    the `from hostlock_gate import require` that `ab.py` uses, so matching
    only the qualified form would report the one driver that has taken the
    lock since #2032 as ungated.

    From the tree rather than the text, and for the same reason the rest of
    this file is: a regex matches a *commented-out* gate call, and that error
    is fail-open -- it reports an unprotected driver as protected, which is
    the one direction this check must never fail in.

    Fails closed on an alias (`g = hostlock_gate.require; g(cmd)`): reported
    ungated, which costs someone an argument rather than costing the box a
    contended run.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "require":
            value = func.value
            if getattr(value, "id", None) == "hostlock_gate":
                return True
        elif isinstance(func, ast.Name) and func.id == "require":
            # The bare name only counts when it came from the gate: a file
            # with a `require()` of its own has not taken the lock.
            if imports_require_from_gate(tree):
                return True
    return False


def imports_require_from_gate(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "hostlock_gate":
            if any(a.name == "require" for a in node.names):
                return True
    return False


# ---------------------------------------------------------------------------
# The second root: crates/onnx-runtime-ep-cpu/benches
# ---------------------------------------------------------------------------
#
# Read by behaviour rather than by declaration -- see the module docstring.
# A file here that starts a process, imports the runtime or opens a session
# must hold the lock or appear below.
EP_BENCHES = ORT_AB.parents[1] / "crates" / "onnx-runtime-ep-cpu" / "benches"

# The two recognised reasons, both recording rather than excusing:
#   known-gap: the file saturates the box and does not hold the lock yet.
#   no-bench:  the file starts a process that is not a benchmark (reading
#              topology out of `lscpu`, say), so the lock would be noise.
#   wrapper:   the file `exec`s whatever it is handed; its *caller* is the
#              harness and holds the lock. A wrapper must not take the lock
#              itself: `exec` keeps the pid and drops the release, so the
#              lock would outlive the run and need reaping.
EP_GAP = "known-gap:"
EP_NOT_A_BENCH = "no-bench:"
EP_WRAPPER = "wrapper:"
EP_REASONS = (EP_GAP, EP_NOT_A_BENCH, EP_WRAPPER)

# Ungated harnesses as of #2043. Every one of these launches a decode or
# prefill run at width 8-16 and can therefore ruin, or be ruined by, anything
# else on the box. They are Roy's and Sebastian's files; this records their
# status without editing them, and each line disappears when its file gates.
EP_LEDGER = {
    name: f"known-gap:#2043 - saturating acc0 harness, owner to gate it ({what})"
    for name, what in (
        ("acc0_lowwidth_smt.py", "t=2 SMT pairing"),
        ("acc0_w16_clock_state.py", "w16 clock/boost state"),
        ("acc0_w16_foreign_load.py", "w16 foreign-load falsifier"),
        ("acc0_w16_mode_placement.py", "w16 mode placement"),
        ("acc0_w16_mode_split.py", "w16 mode split"),
        ("acc0_w16_mode_worker_split.py", "w16 per-component window"),
        ("acc0_w16_page_backing.py", "w16 page-backing lottery"),
        ("acc0_w16_straggler_aslr.py", "w16 straggler vs address layout"),
        ("acc0_w16_straggler_thp.py", "w16 straggler vs THP"),
        ("acc0_worker_placement_probe.py", "starts a run to read worker pins"),
        ("ort_baseline.py", "ORT CPU EP f32 baseline"),
        ("ort_matmulnbits_baseline.py", "ORT MatMulNBits decode baseline"),
        # Found only once the delegation was resolved: these six spawn
        # nothing themselves, they call `acc0_gap_matrix.native` or a wrapper
        # around it. They saturate the box exactly as much as the ones above.
        ("acc0_w16_blocktime_ab.py", "w16 block-time A/B via gap_matrix"),
        ("acc0_w16_straggler_identity.py", "w16 straggler identity"),
        ("acc0_w16_straggler_window.py", "w16 straggler window"),
        ("acc0_w16_study.py", "w16 study, runs native and ORT arms"),
        ("acc0_w8_w16_cpu_split.py", "w8/w16 cpu split"),
        ("acc0_w8_w16_scaling.py", "w8/w16 scaling"),
        # Found only once same-file delegation was resolved: it runs four
        # arms at width 16 through `H.native` and calls nothing else, so a
        # cross-module-only table read it as starting nothing.
        ("acc0_w16_steal_ab.py", "w16 steal-tiles A/B, four arms via H.native"),
    )
}

# Not a gap: `acc0_nothp_exec.py` is a THP-disabling `execvp` wrapper called
# *by* `acc0_w16_straggler_thp.py`. The caller is the harness. Found by the
# spawn detector rather than by me reading the directory, which is the point.
EP_LEDGER["acc0_nothp_exec.py"] = (
    EP_WRAPPER + "execs the command it is given; the caller carries the lock "
    "or the gap -- today that caller is acc0_w16_straggler_thp.py, which has "
    "a gap above"
)

SPAWN_CALLS = {
    "subprocess": {"run", "Popen", "call", "check_call", "check_output"},
    "os": {
        "system",
        "popen",
        "execv",
        "execvp",
        "spawnv",
        "spawnvp",
        "posix_spawn",
        "fork",
        "forkpty",
    },
    # None of these is in the tree today. They are here because the cost of
    # adding a name to a set is nil and the cost of the omission is a
    # saturating harness that reads as harmless -- the direction this check
    # must never fail in.
    "multiprocessing": {"Process", "Pool", "get_context"},
    "concurrent.futures": {"ProcessPoolExecutor"},
}


def spawns_a_process(source: str) -> bool:
    """Starts something. In a benches directory that is the harness signal.

    `looks_like_a_driver` is not enough here: every acc0 harness takes the
    binary as `sys.argv[1]` or builds the path at runtime, so not one of them
    has a `target/release/` literal, and only two import the runtime. Reading
    the spawn instead catches all thirteen.

    Aliased imports count (`from subprocess import run`), because a check a
    rename defeats is a check that reports the tree it wishes it had.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    aliases: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in SPAWN_CALLS:
            for a in node.names:
                if a.name in SPAWN_CALLS[node.module]:
                    aliases[a.asname or a.name] = node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in SPAWN_CALLS:
                    modules[a.asname or a.name] = a.name
                elif a.name.rsplit(".", 1)[0] in SPAWN_CALLS:
                    # `import concurrent.futures` binds `concurrent`, and the
                    # call site reads `concurrent.futures.ProcessPoolExecutor`.
                    modules.setdefault(a.name.split(".")[0], a.name.rsplit(".", 1)[0])
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        path = dotted_name(func)
        if path and "." in path:
            prefix, _, attr = path.rpartition(".")
            module = modules.get(prefix) or (prefix if prefix in SPAWN_CALLS else None)
            if module and attr in SPAWN_CALLS[module]:
                return True
        elif isinstance(func, ast.Name) and func.id in aliases:
            return True
    return False


def dotted_name(node: ast.AST) -> str | None:
    """`concurrent.futures.ProcessPoolExecutor` as a string, or None.

    A single `Name.attr` step is not enough: `import concurrent.futures`
    binds only `concurrent`, so the call site is two attributes deep and a
    one-step match reads it as something else entirely.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def spawning_names(source: str) -> set[str]:
    """Top-level defs and classes whose body starts a process."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if spawns_a_process(ast.unparse(node)):
                out.add(node.name)
    return out


def spawning_helpers(sources: dict[str, str]) -> dict[str, set[str]]:
    return {k: set(v) for k, v in _spawning_helpers(tuple(sorted(sources.items()))).items()}


@functools.lru_cache(maxsize=8)
def _spawning_helpers(items: tuple[tuple[str, str], ...]) -> dict[str, frozenset[str]]:
    """module stem -> the names in it that start a process, directly or not.

    Iterated to a fixpoint, because the delegation is layered and it is
    layered in two directions. Across modules: `acc0_gap_matrix.native` is
    wrapped by `acc0_w16_worker_split`, which six more files call. And
    *within* a module: `native` itself contains no `subprocess` call, it
    calls the same-file helper `sh`. Missing the second kind is not a
    theoretical loss -- it left `acc0_w16_steal_ab.py`, which runs four arms
    at width 16 through `H.native` and nothing else, reading as "starts
    nothing" and needing neither a gate nor an entry. The files that looked
    like they proved the resolution worked were passing on a *different*
    call (`H.ort`), which is why the cell asserting it is now written against
    the table rather than against the verdict.

    Delegating the `subprocess.run` one function or one import away is not a
    way to stop saturating the box.

    Stems that collide (`old/gap_matrix.py` beside `gap_matrix.py`) are
    unioned rather than overwritten. Last-write-wins would let an archived
    copy that spawns nothing empty the live module's entry, and every file
    importing it would read as harmless -- fail-open, from a file nobody
    thought they were changing.
    """
    sources = dict(items)
    table: dict[str, set[str]] = {}
    for name, source in sources.items():
        table.setdefault(Path(name).stem, set()).update(spawning_names(source))
    for _ in range(len(table) + 1):
        grew = False
        for name, source in sources.items():
            stem = Path(name).stem
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if node.name in table[stem]:
                    continue
                if calls_a_spawning_helper(
                    ast.unparse(node), table, imports_from=tree, own_module=stem
                ):
                    table[stem].add(node.name)
                    grew = True
        if not grew:
            break
    return {k: frozenset(v) for k, v in table.items()}


def calls_a_spawning_helper(
    source: str,
    table: dict[str, set[str]],
    imports_from: ast.Module | None = None,
    own_module: str | None = None,
) -> bool:
    """A call into a process-starting helper, here or in a module it imports.

    `imports_from` supplies the import statements when `source` is a fragment
    (one function out of a module), since the imports live at module level.
    `own_module` is that fragment's own module, so a call to a same-file
    helper counts -- the `native` -> `sh` shape above.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    scope = imports_from if imports_from is not None else tree
    aliases: dict[str, str] = {}
    # local name -> (module, name to look up in that module's table). The two
    # differ under `from m import ort as run_ort`, where the table holds the
    # remote name and the call site uses the local one; looking up the local
    # name there silently found nothing.
    direct: dict[str, tuple[str, str]] = {}
    star: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Import):
            for a in node.names:
                stem = a.name.split(".")[0]
                if stem in table:
                    aliases[a.asname or stem] = stem
        elif isinstance(node, ast.ImportFrom):
            stem = (node.module or "").split(".")[0]
            if stem in table:
                for a in node.names:
                    if a.name == "*":
                        star.add(stem)
                    else:
                        direct[a.asname or a.name] = (stem, a.name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = aliases.get(func.value.id)
            if module and func.attr in table[module]:
                return True
        elif isinstance(func, ast.Name):
            found = direct.get(func.id)
            if found and found[1] in table[found[0]]:
                return True
            if any(func.id in table[m] for m in star):
                return True
            if own_module and func.id in table.get(own_module, ()):
                return True
    return False


def uses_the_shell_lock(source: str) -> bool:
    """The other idiom in the tree: `with HostLock(...)`.

    `acc0_gap_matrix.py` implements a lock client that shells out to
    `scripts/hostlock.sh`, and two more harnesses import it as
    `acc0_gap_matrix as H` and use `H.HostLock`. That is genuinely holding
    the lock, so a checker that recognised only `hostlock_gate.require` would
    report three gated harnesses as ungated -- a false alarm in somebody
    else's lane, which is worse than no check, because the answer to it is to
    delete the check.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "HostLock":
                return True
    return False


def holds_the_lock(source: str) -> bool:
    return takes_the_lock(source) or uses_the_shell_lock(source)


def looks_like_a_harness(
    source: str, table: dict[str, set[str]] | None = None
) -> bool:
    if looks_like_a_driver(source) or spawns_a_process(source):
        return True
    return bool(table) and calls_a_spawning_helper(source, table)


def acquired_inside_a_loop(source: str) -> list[int]:
    """Lines where the lock is taken from inside a `for` or `while`.

    The property #1806 was built for and #1803 was filed over: the holder is
    the outer harness spanning every A/B/null arm, not each arm. A lock taken
    per arm releases between them, which is precisely the gap Sebastian
    sampled in when he read the host as clear -- the lock would be *reported*
    on every row and hold across none of the comparison.

    A structural half of custody, not the whole of it: a harness could still
    acquire once and start its arms from a helper called elsewhere. The
    run-time half is the `host_lock` column.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[int] = []

    def walk(node: ast.AST, in_loop: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                name = getattr(child.func, "attr", None) or getattr(
                    child.func, "id", None
                )
                if in_loop and name in ("HostLock", "require"):
                    hits.append(child.lineno)
            # A flag rather than a depth counter: only "inside any loop"
            # is ever read, and a counter invites a mutation that changes
            # the number without changing the answer.
            walk(child, in_loop or isinstance(child, (ast.For, ast.While, ast.AsyncFor)))

    walk(tree, False)
    return sorted(hits)


def ep_sources() -> dict[str, str]:
    if not EP_BENCHES.is_dir():
        return {}
    return {
        p.relative_to(EP_BENCHES).as_posix(): p.read_text()
        for p in sorted(EP_BENCHES.rglob("*.py"))
    }


def ep_failures(
    sources: dict[str, str], ledger: dict[str, str] | None = None
) -> list[str]:
    """Harnesses in the benches root that neither hold the lock nor say why."""
    ledger = EP_LEDGER if ledger is None else ledger
    table = spawning_helpers(sources)
    out = []
    for name, source in sorted(sources.items()):
        if not looks_like_a_harness(source, table):
            continue
        if holds_the_lock(source):
            continue
        reason = ledger.get(name)
        if reason is None:
            out.append(
                f"{name}: starts a benchmark, does not hold the host lock, and "
                "carries no recorded gap"
            )
        elif not reason.startswith(EP_REASONS):
            out.append(
                f"{name}: recorded with an unrecognised reason {reason!r} -- "
                f"expected one of {EP_REASONS}"
            )
    return out


def stale_records(
    sources: dict[str, str], ledger: dict[str, str], prefix: str = EP_GAP
) -> list[str]:
    """Recorded gaps whose file now holds the lock.

    The half that keeps the ledger honest. Without it, gating a harness
    leaves its `known-gap:` line behind, and the record slowly becomes a
    description of the tree as it was -- with the failure mode that a file
    which *stops* gating is then covered by a stale exemption nobody meant to
    grant.
    """
    out = []
    for name, reason in sorted(ledger.items()):
        source = sources.get(name)
        if source is None or not reason.startswith(prefix):
            continue
        if holds_the_lock(source):
            out.append(
                f"{name}: recorded as {prefix.rstrip(':')} but it now holds the "
                "lock -- delete the record"
            )
    return out


def dead_records(
    sources: dict[str, str], ledger: dict[str, str] | None = None
) -> list[str]:
    """Entries naming a file that is gone, or that stopped being a harness."""
    ledger = EP_LEDGER if ledger is None else ledger
    table = spawning_helpers(sources)
    out = []
    for name in sorted(ledger):
        source = sources.get(name)
        if source is None:
            out.append(f"{name}: recorded, but no such file -- delete the record")
        elif not looks_like_a_harness(source, table):
            out.append(
                f"{name}: recorded, but it no longer starts anything -- delete "
                "the record"
            )
    return out


# ---------------------------------------------------------------------------
# The same root, in another language
# ---------------------------------------------------------------------------
#
# `ep_sources` globs `*.py`, so every non-Python harness sitting beside those
# files was never read at all -- not classified, not exempted, invisible. The
# directory holds two shell harnesses and thirteen Rust `[[bench]]` targets.
#
# That is not a corner. `decode_gap_park_ab --bench` is one of the three
# processes that collided on the box in #1803, the incident this whole file
# exists to make visible, and it is a Rust bench. `decode_placement_census.sh`
# carried the sentence "it still runs under the hostlock" in its own header
# while containing no call to `scripts/hostlock.sh` at all -- a declaration
# with nothing behind it, which is the failure mode the ledger was built to
# end, one file extension out of reach.
#
# Shell is covered here. **Rust is not**, and the reason is structural rather
# than an oversight: `cargo bench` compiles on every core for minutes before
# the binary runs, so the thing that must hold the lock is the *invocation*,
# not the target's source. Checking that is a documentation-conformance
# question about the benches README, a different check with a different
# failure mode, and it is deliberately a separate change (#2129) rather than
# bolted on here.
#
# The rule for shell is categorical -- every `*.sh` under the benches root
# holds the lock or carries a recorded reason -- where the Python rule is
# behavioural. That asymmetry is deliberate. `looks_like_a_harness` reads a
# Python AST; the standard library has no shell parser, so the equivalent
# behavioural test would be a regex guessing at intent, and a guess that
# reports somebody else's quiet script as a saturating harness is exactly the
# false alarm that gets a check deleted. A shell file in a benchmark
# directory is a benchmark until its author writes down that it is not, and
# writing that down is one ledger line.
#
# `decode_placement_census.sh` is not here because it now gates. Editing it
# rather than recording it is the one asymmetry with #2043's treatment of
# other people's Python, and the reason is that its header already declared
# the lock -- the change makes the author's own stated intent true and
# deletes a false sentence, rather than imposing a policy on their file.
# `int4_modulo_arms.sh` declares nothing of the kind, so it is recorded.
EP_SHELL_LEDGER: dict[str, str] = {
    "int4_modulo_arms.sh": (
        "known-gap:#2043 - three release builds saturate every core for "
        "minutes, owner to gate it (int4 modulo A/B arms, #1809)"
    ),
}

# `hostlock.sh` subcommands that actually take the lock. `status`,
# `provenance`, `wait` and `release` all name the script without claiming the
# host, and a substring check for "hostlock.sh" would count every one of them
# -- the #2106 lesson, where a custody guard keyed on the word `acquire`
# missed `run`, which acquires too and is the idiom the docs recommend.
SHELL_ACQUIRING = ("run", "acquire")


# Words that may precede a command without making it an argument. `exec ./x`
# and `TMPDIR=/x env ./x` are both still `./x` in command position; `echo ./x`
# is not.
SHELL_PREFIX_WORDS = frozenset(
    {
        "exec",
        "command",
        "builtin",
        "sudo",
        "env",
        "time",
        "nice",
        "setsid",
        "nohup",
        # Keywords a command can follow directly.
        "then",
        "else",
        "do",
    }
)

# The heredoc operator, and only that. `(?<!<)` and `(?!<)` keep a `<<<`
# herestring out, which would otherwise register a phantom heredoc.
# `<<-` is captured rather than matched away, because the two forms terminate
# differently: `<<` wants the delimiter alone on the line, `<<-` allows
# leading **tabs** and nothing else.
#
# The delimiter is captured whole, as the *word* it is: one run of `\x`
# escapes, quoted segments and ordinary characters. `\w+` was a silent
# exemption -- `<<'EO-F'` captured `EO`, whose terminator never arrives, so
# the body (usage text with a `hostlock.sh run` in it) was read as code --
# and matching only a fully quoted *or* fully bare delimiter was the same
# mistake one notch narrower: `<<\EOF` captured `\EOF` and `<<E"O"F`
# captured `E`, neither of which is the terminator the shell will look for.
# Those two fail *closed* rather than open, but `<<\EOF` is a perfectly
# ordinary way to write `<<'EOF'`, and a check that demands a lock from a
# file that already takes one gets deleted rather than obeyed.
_HEREDOC = re.compile(
    r"(?<!<)(<<-?)(?!<)\s*((?:\\.|'[^']*'|\"[^\"]*\"|[^\s;&|<>()])+)"
)

# The word after the operator, in the same shape, used to decide which quoted
# regions have to survive into the heredoc scan: the delimiter's own quotes
# do, everything else is blanked so a `<<` inside a string cannot open a
# body. Anchored at the end and searched from the start of the line rather
# than through a fixed lookback -- `\s*` is unbounded, so any fixed window is
# defeated by enough whitespace -- and it spans a partial word so the inner
# quotes of `<<E"O"F` are kept too.
_HEREDOC_OPEN = re.compile(
    r"<<-?\s*(?:\\.|'[^']*'|\"[^\"]*\"|[^\s;&|<>()])*$"
)


def _heredoc_delimiter(word: str) -> str:
    """The terminator `sh` will compare against, from the word as written.

    Quote removal, which the shell does before it ever looks for the
    terminator: `'EOF'`, `"EOF"`, `\\EOF` and `E"O"F` all name `EOF`. Doing
    this rather than special-casing a wholly quoted delimiter is what keeps
    the two mixed forms from blanking to end of file and swallowing a real
    acquisition below them.
    """
    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch == "\\" and i + 1 < n:
            out.append(word[i + 1])
            i += 2
        elif ch in "'\"":
            close = word.find(ch, i + 1)
            if close < 0:
                out.append(word[i + 1 :])
                break
            out.append(word[i + 1 : close])
            i = close + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)

# Arithmetic is not a redirection. `$(( 1 << k ))` and `(( n = 1 << k ))`
# would otherwise open a heredoc whose delimiter is the shift's right-hand
# word -- and since an unterminated heredoc blanks to end of file, that would
# swallow a real acquisition below it and report a gated harness as ungated.
_ARITH = re.compile(r"\$\(\(|\(\(")

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# The script named as a word: optionally a path, bounded on the left by a
# shell separator or a quote, so `my-hostlock.sh` is not a match and
# `"$ROOT/scripts/hostlock.sh"` is.
_MENTION = re.compile(
    "(?:^|[\\s;|&(){}`\"'])((?:[^\\s;|&(){}`\"']*/)?hostlock\\.sh)\\b"
)

# Quotes and separators become whitespace when the *subcommand* is read, so
# `hostlock.sh "run"` and `hostlock.sh acquire; done` both yield their verb.
_TO_SPACE = str.maketrans({c: " " for c in "\"';|&(){}`"})

# Characters after which a `#` starts a comment, and after which a word is in
# command position. Both are the shell's own metacharacters plus whitespace.
SHELL_SEPARATORS = " \t\n;|&(){}`"


def _heredoc_body_end(source: str, start: int, dash: bool, delim: str) -> int:
    """Where the body opened by `<<delim` ends -- end of file if it never does.

    The terminator must be the delimiter **alone** on its line -- `<<-` also
    allows leading tabs, and nothing else does. Matching it with `.strip()`
    was fail-open: an indented `  EOF` inside the body ended the heredoc
    early and exposed the rest of it as code, which for a usage block is
    precisely the false custody this scanner exists to refuse.

    A delimiter that never arrives consumes the rest of the file, which is
    what `sh` does: the body is fed to the command, and a warning is all
    `bash` says about it (`dash` says nothing). The file *runs*; what it does
    not do is execute its heredoc. Blanking to end of file is therefore the
    faithful reading, and it is also the fail-closed one -- the cost is a
    ledger line if this scanner ever mistakes something else for a heredoc
    operator, which is why `<<<`, arithmetic and quoted `<<` are excluded
    before we get here rather than after.
    """
    n = len(source)
    probe = start
    while probe < n:
        end = source.find("\n", probe)
        end = n if end < 0 else end
        line = source[probe:end].rstrip("\r")
        if (line.lstrip("\t") if dash else line) == delim:
            return end
        probe = end + 1
    return n


def strip_shell_comments(source: str) -> str:
    """Blank out comments, heredoc bodies and multi-line quoted strings.

    Blanked rather than deleted so that offsets and "the rest of this logical
    line" survive. There is no shell parser in the standard library, so this
    is a scanner rather than a parse, and what it is for is narrow: after it
    runs, together with the command-position test in [`shell_holds_the_lock`],
    a `hostlock.sh` left in the text is one the shell would actually execute
    rather than one it would print.

    The shapes it exists to remove are the ones that made the first version
    of this **fail open**, which is the direction that matters here: a
    categorical rule exempts whatever it reads as gated, so a false positive
    is a silent pass with no ledger line -- the same "declaration with
    nothing behind it" this whole check was written to end.

      * `cat <<EOF ... hostlock.sh run ... EOF`, including `<<'EOF'` and
        `<<-EOT` -- usage text is an entirely plausible thing for a bench
        script to print, and it would have read as custody.
      * `echo hi;# scripts/hostlock.sh run -- x`, `echo "it's x" # ...` and
        `echo '"' # ...` -- real comments that the first version's
        "preceded by a space, even quote count" heuristic kept.
      * A quoted string spanning a newline. Its interior newlines are
        replaced by **spaces** rather than kept, so what follows the closing
        quote stays on the same logical line: with the newlines kept,
        `printf 'a\\nuse: %s %s\\n' scripts/hostlock.sh run` put the trailing
        arguments at the start of a line of their own, where they read as a
        command. That was a fail-open, and it is why this is not simply
        "blank the region".

    Single-line quoted regions are kept verbatim, because a quoted word is
    often the command itself -- `"$ROOT/scripts/hostlock.sh" run` -- and
    blanking those would lose a real acquisition. Deciding between a quoted
    *path* and quoted *prose* is the command-position test's job, not this
    one's. `$'...'` processes `\\'`, and is scanned that way.

    `${x#-}`, `$#` and `a#b` are not comments and survive intact, because a
    `#` only opens one at the start of a word.

    The heredoc scan runs over a **second** blanking of the same text in
    which every quoted region is blanked, so `echo "x << 2"` cannot register
    a heredoc; the sole exception is a quoted word directly after the
    operator, which is the delimiter (`<<'EOF'`). Arithmetic on one line
    (`$(( 1 << k ))`, `(( n = 1 << k ))`) is blanked there too, because a
    shift is not a redirection and its right-hand word is not a delimiter.
    A heredoc whose delimiter never arrives consumes the rest of the file,
    which is what `sh` does with it.

    What it does not model, each with the direction it fails in:

      * Deliberate indirection -- `eval`, a subcommand held in a variable, a
        `$LOCK` alias. **Fail-open**, exactly as the Python side cannot see
        `getattr(subprocess, "run")`, and not closable by reading source.
      * A `<<` this scanner still mistakes for a heredoc operator -- one
        inside a construct not modelled above -- whose delimiter then never
        arrives, blanking the rest of the file. **Fail-closed**: it costs a
        ledger line in somebody else's lane rather than a silent pass, which
        is the direction to fail in but not a free one, because a check that
        cries wolf gets deleted rather than obeyed.
    """
    out: list[str] = []
    # A parallel copy in which *every* quoted region is blanked. `out` keeps
    # single-line quotes because they are often the command path; the heredoc
    # scan must not, or a `<<` inside a string opens a phantom body.
    scan: list[str] = []
    pending: list[tuple[bool, str]] = []
    chunk = 0
    i, n = 0, len(source)

    def emit(text: str) -> None:
        out.append(text)
        scan.append(text)

    def close_line() -> None:
        """Blank the bodies of any heredocs the finished line opened."""
        nonlocal i, chunk
        for m in _HEREDOC.finditer("".join(scan[chunk:])):
            pending.append(
                (m.group(1).endswith("-"), _heredoc_delimiter(m.group(2)))
            )
        i += 1
        while pending and i < n:
            dash, delim = pending.pop(0)
            end = _heredoc_body_end(source, i, dash, delim)
            emit("".join(c if c == "\n" else " " for c in source[i:end]))
            i = end
            if i < n:
                emit("\n")
                i += 1
        chunk = len(out)

    while i < n:
        ch = source[i]
        if ch == "\\" and i + 1 < n:
            emit(source[i : i + 2])
            i += 2
            continue
        arith = _ARITH.match(source, i)
        if arith:
            # Arithmetic, kept in `out` (it is code) and blanked in `scan`
            # (it is not a redirection). Only when the whole thing sits on
            # one line: a `((` spanning lines is far more likely to be two
            # subshells, and blanking those in the scan would hide a real
            # heredoc opened inside them.
            depth, probe = 0, arith.start()
            while probe < n:
                if source[probe] == "(":
                    depth += 1
                elif source[probe] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif source[probe] == "\n":
                    break
                probe += 1
            if probe < n and depth == 0:
                region = source[i : probe + 1]
                out.append(region)
                scan.append(" " * len(region))
                i = probe + 1
                continue
        if ch in "'\"":
            # `$'...'` is the one single-quoted form that processes `\'`.
            escapes = ch == '"' or source[i - 1 : i] == "$"
            close = i + 1
            while close < n and source[close] != ch:
                close += 2 if (escapes and source[close] == "\\") else 1
            close = min(close, n - 1)
            region = source[i : close + 1]
            if "\n" in region:
                out.append(" " * len(region))
            else:
                out.append(region)
            scan.append(
                region
                if _HEREDOC_OPEN.search(source[source.rfind("\n", 0, i) + 1 : i])
                else " " * len(region)
            )
            i = close + 1
            continue
        if ch == "#" and (i == 0 or source[i - 1] in SHELL_SEPARATORS):
            end = source.find("\n", i)
            end = n if end < 0 else end
            emit(" " * (end - i))
            i = end
            continue
        if ch == "\n":
            emit("\n")
            close_line()
            continue
        emit(ch)
        i += 1
    return "".join(out)


def _in_command_position(line: str, start: int) -> bool:
    """Is the word beginning at `start` the command, not one of its arguments?

    Walks left over whitespace, over the prefix words that keep a command a
    command (`exec`, `env`, ...) and over `VAR=value` assignments. Anything
    else in front of it -- `echo`, `printf`, a `--reason` -- means the name
    is being *mentioned*, not run.
    """
    head = line[:start]
    while True:
        stripped = head.rstrip().rstrip("\"'")
        if stripped != head:
            head = stripped
            continue
        if not head or head[-1] in ";|&(){}`":
            return True
        word = head.rsplit(None, 1)[-1] if head.split() else ""
        if word in SHELL_PREFIX_WORDS or _ASSIGNMENT.match(word):
            head = head[: len(head) - len(word)]
            continue
        return False


def shell_holds_the_lock(source: str) -> bool:
    """The script *runs* `scripts/hostlock.sh` with an acquiring subcommand.

    Three conditions, and each one is a defect that was found rather than
    imagined:

    * The name survives comment/string/heredoc blanking, so usage text that
      tells the reader to run under the lock is not custody (fail-open).
    * The name is in command position, so `echo scripts/hostlock.sh` and
      `LOCK=scripts/hostlock.sh` are not custody either (fail-open).
    * The subcommand -- the next word **on the same logical line** -- is one
      that takes the lock. `status`, `provenance`, `wait` and `release` all
      name it without claiming the host (#2106's lesson), and reading past
      the end of the line let an unrelated `run` on the next line count.

    Line continuations are joined first, so a backslash before the
    subcommand is still the same logical line.
    """
    code = strip_shell_comments(source).replace("\\\n", "  ")
    for line in code.splitlines():
        for m in re.finditer(_MENTION, line):
            start = m.start(1)
            if not _in_command_position(line, start):
                continue
            rest = line[m.end(1) :].translate(_TO_SPACE)
            word = next(iter(rest.split()), "")
            if word in SHELL_ACQUIRING:
                return True
    return False


def ep_shell_sources() -> dict[str, str]:
    if not EP_BENCHES.is_dir():
        return {}
    return {
        p.relative_to(EP_BENCHES).as_posix(): p.read_text()
        for p in sorted(EP_BENCHES.rglob("*.sh"))
    }


def ep_shell_failures(
    sources: dict[str, str], ledger: dict[str, str] | None = None
) -> list[str]:
    """Shell files in the benches root that neither gate nor say why."""
    ledger = EP_SHELL_LEDGER if ledger is None else ledger
    out = []
    for name, source in sorted(sources.items()):
        if shell_holds_the_lock(source):
            continue
        reason = ledger.get(name)
        if reason is None:
            out.append(
                f"{name}: shell harness in the benches root, does not hold the "
                "host lock, and carries no recorded reason"
            )
        elif not reason.startswith(EP_REASONS):
            out.append(
                f"{name}: recorded with an unrecognised reason {reason!r} -- "
                f"expected one of {EP_REASONS}"
            )
    return out


def dead_shell_records(
    sources: dict[str, str], ledger: dict[str, str] | None = None
) -> list[str]:
    """Shell entries naming a file that is gone."""
    ledger = EP_SHELL_LEDGER if ledger is None else ledger
    return [
        f"{name}: recorded, but no such file -- delete the record"
        for name in sorted(ledger)
        if name not in sources
    ]


def stale_shell_records(
    sources: dict[str, str], ledger: dict[str, str] | None = None
) -> list[str]:
    """Shell gaps whose file now gates -- the record has to go with the fix."""
    ledger = EP_SHELL_LEDGER if ledger is None else ledger
    out = []
    for name, reason in sorted(ledger.items()):
        source = sources.get(name)
        if source is None or not reason.startswith(EP_GAP):
            continue
        if shell_holds_the_lock(source):
            out.append(
                f"{name}: recorded as known-gap but it now holds the lock -- "
                "delete the record"
            )
    return out


# The third language in that directory, and the one this file does not yet
# check: thirteen `[[bench]]` targets in `crates/onnx-runtime-ep-cpu/`.
# Deferred to #2129 rather than bodged, because what must hold the lock for a
# Rust bench is the `cargo bench` *invocation* -- the source has no process
# boundary to wrap, and `cargo bench -p onnx-runtime-ep-cpu` (the headline
# line in that directory's README) runs all thirteen back to back.
#
# Pinning the name set is not a substitute for gating them. It is the cheap
# half: while #2129 is open, adding a Rust bench fails here, so the decision
# to leave it uncovered is taken consciously by whoever adds it instead of
# defaulting to silence -- which is the exact way `decode_gap_park_ab` came to
# be one of the three processes in the collision that motivated #1803.
EP_RUST_UNCOVERED = frozenset(
    {
        "activation_bench",
        "decode_gap_park_ab",
        "gqa_decode",
        "half_decode_gemv_ab",
        "half_prefill_route_ab",
        "int4_acc0_attribution",
        "int4_decode_loop_ab",
        "int4_prefill_route_ab",
        "int8_prefill_route_ab",
        "kernels",
        "matmul_nbits_prefill_ab",
        "native_vs_mlas",
        "sdpa_simd",
    }
)

_BENCH_NAME = re.compile(r"""^\s*name\s*=\s*['"]([^'"]+)['"]""")


@functools.lru_cache(maxsize=1)
def ep_rust_benches() -> frozenset[str]:
    """`[[bench]]` target names in the EP crate's manifest."""
    manifest = EP_BENCHES.parent / "Cargo.toml"
    names, in_bench = set(), False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            # `[ [bench] ]` is legal TOML and would otherwise read as a
            # different table, silently dropping its target from the pin.
            in_bench = "".join(stripped.split()) == "[[bench]]"
            continue
        if in_bench:
            found = _BENCH_NAME.match(line)
            if found:
                names.add(found.group(1))
    return frozenset(names)


def declared_roles(
    names: list[str], lists: dict[str, dict[str, str]] | None = None
) -> dict[str, list[str]]:
    """Which role each file was declared as. Zero or two is the finding."""
    lists = lists or {
        "driver": DRIVERS,
        "library": LIBRARIES,
        "test": TESTS,
        "generator": GENERATORS,
    }
    return {
        name: [role for role, members in lists.items() if name in members]
        for name in names
    }


def unclassified(
    names: list[str], lists: dict[str, dict[str, str]] | None = None
) -> list[str]:
    roles = declared_roles(names, lists)
    return sorted(n for n, found in roles.items() if len(found) != 1)


def looks_like_a_driver(source: str, binary_paths: bool = True) -> bool:
    """Behaviour, not vocabulary.

    `binary_paths=False` drops the string-constant rule, which is how the test
    files are read: a path to a built binary is *behaviour* in a harness and
    *data* in a test whose fixtures describe harnesses. Imports and calls are
    still checked there, because those execute wherever they appear.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in RUNTIME_MODULES for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in RUNTIME_MODULES:
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in SESSION_CALLS:
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A path into `target/` is a built binary being started, which is
            # the shape of every native harness here. Inside a docstring it is
            # prose, so the module docstring is skipped below.
            if (
                binary_paths
                and BUILT_BINARY.search(node.value)
                and node is not first_docstring(tree)
            ):
                return True
    return False


def first_docstring(tree: ast.Module) -> ast.AST | None:
    body = getattr(tree, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return body[0].value
    return None


def gate_failures(sources: dict[str, str]) -> list[str]:
    """Drivers that neither take the lock nor carry a recorded gap.

    Reads the source rather than importing it: a driver that has to be
    imported to be checked is a driver that runs its arguments parser, and
    half of these need a model on disk.
    """
    out = []
    for name, source in sorted(sources.items()):
        reason = DRIVERS.get(name)
        if reason is None:
            continue
        if takes_the_lock(source):
            continue
        if reason.startswith("known-gap:"):
            continue
        out.append(f"{name}: declared a driver, never calls hostlock_gate.require")
    return out


def contradicted(sources: dict[str, str]) -> list[str]:
    """Files declared harmless that behave like drivers.

    This is the half that matters in a year: the classification above is a
    claim about each file, and a claim nothing checks is a label that drifts.
    """
    out = []
    for name, source in sorted(sources.items()):
        if name in DRIVERS:
            continue
        if declared_reason(name).startswith(RUNTIME_OK):
            continue
        if looks_like_a_driver(source, binary_paths=name not in TESTS):
            out.append(
                f"{name}: declared not-a-driver, but it starts a benchmark or "
                "opens an inference session"
            )
    return out


def declared_reason(name: str) -> str:
    for members in (DRIVERS, LIBRARIES, TESTS, GENERATORS):
        if name in members:
            return members[name]
    return ""


def read_sources() -> dict[str, str]:
    """Recursive, because the workflow filter is.

    `paths: scripts/ort_ab/**` triggers the job for
    `scripts/ort_ab/cuda/decode_bench.py`, so a non-recursive glob here would
    run the check, see nothing, and go green -- the vacuity failure this file
    warns about, one directory down.
    """
    return {
        p.relative_to(ORT_AB).as_posix(): p.read_text()
        for p in sorted(ORT_AB.rglob("*.py"))
    }


class Classification(unittest.TestCase):
    def test_every_file_in_the_directory_has_exactly_one_role(self):
        # A new harness lands here as an unclassified file, which fails --
        # rather than as a silently ungated one, which does not.
        self.assertEqual(unclassified(sorted(read_sources())), [])

    def test_a_driver_hidden_one_directory_down_is_still_seen(self):
        # The workflow filter is `scripts/ort_ab/**`, so a push touching
        # `cuda/decode_bench.py` runs this job. A non-recursive glob would
        # have run it, seen nothing and gone green -- the cheapest bypass of
        # all, needing no edit to any list here.
        self.assertEqual(
            unclassified(["cuda/decode_bench.py"]), ["cuda/decode_bench.py"]
        )
        # And discovery really descends: a pure-function assertion would
        # pass just as well with a non-recursive glob, since the directory
        # has no subdirectory today. This one puts a file there.
        planted = ORT_AB / "cuda_probe_tmp" / "decode_bench.py"
        planted.parent.mkdir(parents=True, exist_ok=True)
        try:
            planted.write_text("import onnxruntime\n")
            found = read_sources()
            self.assertIn("cuda_probe_tmp/decode_bench.py", found)
            self.assertEqual(
                unclassified(sorted(found)), ["cuda_probe_tmp/decode_bench.py"]
            )
        finally:
            planted.unlink(missing_ok=True)
            planted.parent.rmdir()

    def test_a_new_unclassified_file_is_a_failure(self):
        self.assertEqual(unclassified(["brand_new_sweep.py"]), ["brand_new_sweep.py"])

    def test_a_file_declared_twice_is_also_a_failure(self):
        # Two roles is not "more documented", it is a disagreement about who
        # has to take the lock.
        both = {"driver": {"x.py": "gated"}, "generator": {"x.py": "fixtures"}}
        self.assertEqual(unclassified(["x.py"], both), ["x.py"])
        self.assertEqual(declared_roles(["ab.py"])["ab.py"], ["driver"])


class GateReuse(unittest.TestCase):
    """The recipe for closing one of those nineteen gaps has to still work.

    The ledger asks nineteen files to gate. If gating means writing a lock
    client, we get nineteen subtly different ones -- which is how this root
    came to have two already. So `hostlock_gate` is usable from another root
    with a path insert, and the three properties that makes true are pinned
    here rather than left to the first person who tries it.
    """

    GATE = ORT_AB / "hostlock_gate.py"
    ACQUIRING = {"acquire", "run"}

    def module_level_names(self) -> set[str]:
        tree = ast.parse(self.GATE.read_text())
        return {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        } | {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }

    def documented_recipe(self) -> str:
        """The fenced snippet under the README's "Closing a gap" heading.

        Extracted rather than restated: a copy here would drift from the copy
        a reader follows, and the copy a reader follows is the one that has
        to work.
        """
        readme = (ORT_AB / "README.md").read_text()
        after = readme.split("### Closing a gap", 1)
        self.assertEqual(len(after), 2, "the README section was renamed or removed")
        block = re.search(r"```python\n(.*?)```", after[1], re.S)
        self.assertIsNotNone(block, "no python block under that heading")
        return block.group(1)

    def test_the_documented_recipe_names_only_things_that_exist(self):
        # Doc drift with a safety instruction in it is worse than no doc: the
        # reader concludes the gate is broken and writes their own client,
        # which is the outcome the shared module exists to prevent.
        docs = self.GATE.read_text() + (ORT_AB / "README.md").read_text()
        # `(?!py\b)` so the filename `hostlock_gate.py` is not read as a call.
        # The capture is deliberately not `[a-z_]+`: `HOSTLOCK` is public and
        # a name this cannot see is a name it cannot check.
        named = set(re.findall(r"hostlock_gate\.(?!py\b)([A-Za-z_][A-Za-z0-9_]*)", docs))
        for group in re.findall(r"from hostlock_gate import ([^\n]+)", docs):
            named |= {n.strip() for n in group.split(",") if n.strip().isidentifier()}
        self.assertTrue(named, "the recipe named nothing -- did the docs move?")
        missing = sorted(named - self.module_level_names())
        self.assertEqual(missing, [])
        # An alias would put every later reference out of the regex's reach
        # and this check back to passing vacuously, so the docs may not use
        # one. Cheaper than resolving it, and there is no reason to want one.
        self.assertNotIn("import hostlock_gate as", docs)

    def test_the_documented_recipe_runs_from_that_root(self):
        # Executed, not parsed. The previous form of this cell read the path
        # depth out of the recipe and checked the arithmetic, which pins the
        # constant that happens to be written rather than the property the
        # reader needs -- and said nothing about a harness one directory
        # deeper, where a fixed depth silently points at `crates/`.
        recipe = self.documented_recipe()
        insert = recipe.split("import hostlock_gate", 1)[0] + "import hostlock_gate\n"
        self.assertIn("scripts", insert)
        self.assertNotRegex(
            insert,
            r"parents\[\d+\]",
            "a fixed depth is correct only for a direct child of that root",
        )
        probe = insert + "print(hostlock_gate.HOSTLOCK.is_file())\n"
        nested = EP_BENCHES / "leon_recipe_probe"
        planted = [EP_BENCHES / "_recipe_probe.py", nested / "_recipe_probe.py"]
        # A decoy `scripts/` between the harness and the repo root. An ascent
        # that stops at the first ancestor merely *called* `scripts` lands
        # here, inserts a path with no gate in it, and the harness dies on
        # `import hostlock_gate` -- so the recipe has to key on the directory
        # it actually needs. Nothing in the tree has one today, which is why
        # this cell has to make one rather than wait for it.
        decoy = EP_BENCHES.parent / "scripts"
        decoy_is_ours = not decoy.exists()
        try:
            nested.mkdir(exist_ok=True)
            if decoy_is_ours:
                decoy.mkdir()
            for path in planted:
                path.write_text(probe)
                out = subprocess.run(
                    [sys.executable, str(path)], capture_output=True, text=True
                )
                self.assertEqual(
                    out.stdout.strip(),
                    "True",
                    f"the documented recipe fails from {path}: {out.stderr[-400:]}",
                )
        finally:
            for path in planted:
                path.unlink(missing_ok=True)
            if nested.is_dir():
                nested.rmdir()
            if decoy_is_ours and decoy.is_dir():
                decoy.rmdir()

    def test_the_gate_does_not_read_the_working_directory(self):
        # The property that makes reuse possible at all, checked by running
        # it from somewhere else entirely: `hostlock.sh` is resolved from the
        # module's own file, so a harness invoked from any directory finds
        # the same lock. A CWD-relative path would work in every test here
        # and fail for every caller in that other root.
        probe = (
            "import sys; sys.path.insert(0, %r)\n"
            "import hostlock_gate as g\n"
            "print(g.HOSTLOCK.is_file())\n" % str(ORT_AB)
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=str(ORT_AB.parents[1].anchor),
        )
        self.assertEqual(out.stdout.strip(), "True", out.stderr[-400:])

    def test_the_gate_never_runs_an_acquiring_subcommand(self):
        # A driver that takes the lock itself releases it when it exits, so a
        # matrix run as several processes is certified arm by arm and
        # protected across none of them -- #1803's mechanism. The gate must
        # therefore only ever *read* the lock.
        #
        # Read from the AST rather than by searching the text for `acquire`:
        # `hostlock.sh run -- CMD` acquires too, and is the more likely
        # regression precisely because it is the idiom the docs recommend to
        # everyone else. A substring check for the wrong word is a guard that
        # reports the defect it was not written for.
        tree = ast.parse(self.GATE.read_text())
        found = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            argv = node.args[0] if node.args else None
            if argv is None or "HOSTLOCK" not in ast.unparse(argv):
                continue
            # Keyed on the argv, not on the callee: `read_provenance` shells
            # out through an injected `runner=subprocess.run` so a test can
            # substitute it, and a check that only knew `subprocess.run` by
            # name would find nothing here and say so by passing.
            found += 1
            if not isinstance(argv, ast.List):
                # `str(HOSTLOCK)` converts the path; anything else that hands
                # the lock script somewhere this cannot read is a refusal,
                # not a pass.
                self.assertIn(
                    dotted_name(node.func),
                    {"str", "os.fspath"},
                    f"unreadable argv at line {node.lineno}",
                )
                found -= 1
                continue
            words = set()
            for element in argv.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    words.add(element.value)
                elif isinstance(element, ast.Call) and dotted_name(element.func) == "str":
                    continue  # `str(HOSTLOCK)`, the path itself
                else:
                    self.fail(f"argv element not readable at line {node.lineno}")
            taking = sorted(words & self.ACQUIRING)
            self.assertEqual(taking, [], f"the gate takes the lock at {node.lineno}")
        # Without this the cell passes by finding nothing -- which is exactly
        # what a rewrite that shells out some other way would produce.
        self.assertTrue(found, "no hostlock.sh invocation found in the gate")

    def test_the_remedy_points_at_the_outer_wrapper(self):
        # The message half of the same property: refusing is only useful if
        # what it prints is the thing that would have worked.
        self.assertIn("hostlock.sh run --owner", self.GATE.read_text())


class Vacuity(unittest.TestCase):
    """A check that does not run for new files cannot catch a new file.

    The workflow's `paths:` filter used to name the four lock files by hand,
    so `scripts/ort_ab/brand_new_sweep.py` would have skipped this job
    entirely -- the one input it exists for.
    """

    WORKFLOW = ORT_AB.parents[1] / ".github" / "workflows" / "hostlock.yml"

    def test_the_workflow_runs_for_any_file_in_this_directory(self):
        # Placement, not multiplicity: two copies of the filter both sitting
        # under `pull_request` would count 2 and pass, while a direct push to
        # main landed an ungated driver unchecked -- which is the thing the
        # second copy is there to prevent.
        pr_block, push_block = self.trigger_blocks()
        self.assertEqual(pr_block.count('"scripts/ort_ab/**"'), 1)
        self.assertEqual(push_block.count('"scripts/ort_ab/**"'), 1)

    def test_the_workflow_runs_for_the_benches_root_too(self):
        # The #2079 finding, one root over: a check that does not run for a
        # new file cannot catch a new file, and the benches root is where the
        # next ungated harness is most likely to land -- 19 of the 23 files
        # there that start a benchmark are ungated today.
        pr_block, push_block = self.trigger_blocks()
        for block in (pr_block, push_block):
            self.assertEqual(block.count('"crates/onnx-runtime-ep-cpu/benches/**"'), 1)

    def test_the_scanned_roots_and_the_trigger_agree(self):
        # Stated as a relation rather than two literals: the roots this file
        # reads must each appear in the filter. Adding a third root without
        # its path fails here instead of going green over an unread tree.
        text = self.WORKFLOW.read_text()
        repo = ORT_AB.parents[1]
        for root in (ORT_AB, EP_BENCHES):
            self.assertIn(f'"{root.relative_to(repo).as_posix()}/**"', text)

    def trigger_blocks(self) -> tuple[str, str]:
        """The `pull_request:` and `push:` halves of the trigger section.

        Split on the text rather than parsed, because pyyaml is not in this
        job's environment and adding a dependency to a conformance check is a
        way to have it skipped.
        """
        text = self.WORKFLOW.read_text()
        body = text.split("\non:\n", 1)[1]
        head, _, tail = body.partition("\n  push:\n")
        self.assertTrue(tail, "workflow has no push: trigger")
        return head, tail.split("\npermissions:", 1)[0]

    def test_the_workflow_actually_runs_this_file(self):
        self.assertIn("test_gate_conformance.py", self.WORKFLOW.read_text())


class Gating(unittest.TestCase):
    def test_every_declared_driver_in_the_tree_gates_or_records_its_gap(self):
        self.assertEqual(gate_failures(read_sources()), [])

    def test_a_driver_that_forgot_the_gate_is_caught(self):
        fail = gate_failures({"ab.py": "import subprocess\nbench_generic\n"})
        self.assertEqual(len(fail), 1)
        self.assertIn("never calls hostlock_gate.require", fail[0])

    def test_a_recorded_gap_is_not_a_pass_by_accident(self):
        # The CUDA driver is ungated on purpose and says so in DRIVERS. The
        # distinction being pinned: `known-gap:` suppresses the failure, an
        # empty or missing reason does not.
        self.assertTrue(DRIVERS["ort_cuda_decode_bench.py"].startswith("known-gap:#"))
        self.assertEqual(gate_failures({"ort_cuda_decode_bench.py": "no gate here"}), [])
        self.assertEqual(
            len(gate_failures({"sweep_decode.py": "no gate here"})),
            1,
        )

    def test_the_gate_call_is_matched_on_the_call_not_the_word(self):
        # A file that mentions the gate in a comment has not taken the lock.
        fail = gate_failures({"ab.py": "# see hostlock_gate for the admission rules\n"})
        self.assertEqual(len(fail), 1)

    def test_a_commented_out_gate_call_is_not_a_gate_call(self):
        # The fail-open direction, and the reason this reads the tree: a
        # regex for `hostlock_gate.require(` matches inside a comment, so a
        # driver whose only gate call is commented out would have reported as
        # protected -- the exact "reads as gated while ungated" failure this
        # file exists to prevent.
        for source in (
            "# hostlock_gate.require(cmd)\n",
            '"""Call hostlock_gate.require(cmd) before any arm."""\n',
        ):
            self.assertFalse(takes_the_lock(source), source)
            self.assertEqual(len(gate_failures({"ab.py": source})), 1)

    def test_a_local_function_called_require_is_not_the_gate(self):
        # The bare name counts only when it came from the gate's module.
        source = "def require(x):\n    pass\n\nrequire(1)\n"
        self.assertFalse(takes_the_lock(source))
        self.assertTrue(
            takes_the_lock("from hostlock_gate import require\nrequire(1)\n")
        )

    def test_importing_the_gate_without_calling_it_is_not_gating(self):
        # The failure mode this exists for: a driver that imports `require`,
        # never calls it, and reads as protected to anything grepping for the
        # module name.
        fail = gate_failures({"ab.py": "from hostlock_gate import require\n"})
        self.assertEqual(len(fail), 1)

    def test_both_import_styles_count_as_gated(self):
        # `ab.py` uses the from-import; `sweep_decode.py` the qualified call.
        self.assertEqual(
            gate_failures(
                {
                    "ab.py": "from hostlock_gate import require\nrequire(cmd)\n",
                    "sweep_decode.py": "hostlock_gate.require(cmd)\n",
                }
            ),
            [],
        )


class Contradiction(unittest.TestCase):
    def test_nothing_declared_harmless_behaves_like_a_driver(self):
        self.assertEqual(contradicted(read_sources()), [])

    def test_a_generator_that_grows_a_benchmark_is_caught(self):
        fail = contradicted(
            {"gen_moe.py": 'subprocess.run(["target/release/bench_generic"])\n'}
        )
        self.assertEqual(len(fail), 1)
        self.assertIn("gen_moe.py", fail[0])

    def test_prose_about_benchmarks_is_not_a_benchmark(self):
        # The #2043 miscount, pinned: `gen_gqa.py` was called a harness for
        # describing one in its docstring.
        self.assertEqual(
            contradicted(
                {"gen_gqa.py": '"""Fixtures for target/release/bench_generic."""\n'}
            ),
            [],
        )

    def test_a_generator_that_opens_an_inference_session_is_caught(self):
        self.assertEqual(
            len(contradicted({"gen_gqa.py": "s = InferenceSession(path)\n"})), 1
        )

    def test_a_test_files_fixtures_are_data_but_its_imports_are_not(self):
        # The line this draws: a test may *describe* a harness (its fixtures
        # are strings naming binaries), but a test that imports the runtime
        # and opens a session is running one.
        fixture = 'CMD = "target/release/bench_generic"\n'
        self.assertEqual(contradicted({"test_ab_lock.py": fixture}), [])
        self.assertEqual(len(contradicted({"test_ab_lock.py": "import onnxruntime\n"})), 1)
        self.assertEqual(len(contradicted({"gen_moe.py": fixture})), 1)

    def test_a_file_that_will_not_parse_is_not_silently_cleared(self):
        # It is cleared -- and that is a real limitation, so it is written
        # down here rather than discovered later. A syntactically broken file
        # cannot run either, so it cannot saturate anything.
        self.assertFalse(looks_like_a_driver("def f(:\n"))

    def test_importing_onnxruntime_counts_only_as_an_import(self):
        # `import onnxruntime` at module level is how every real driver here
        # measures; a generator that only builds graphs uses `onnx`, not
        # `onnxruntime`, so the distinction is load-bearing rather than
        # stylistic.
        self.assertEqual(contradicted({"gen_gemm.py": "import onnx\n"}), [])
        self.assertEqual(
            len(contradicted({"gen_gemm.py": "import onnxruntime as ort\n"})), 1
        )

    def test_a_generator_may_declare_that_it_loads_the_runtime(self):
        # The asymmetry the reviewer caught: drivers could record a gap, but a
        # generator that legitimately checks its fixture loads in ORT had no
        # way to say so and would have been failed into not doing it.
        self.assertTrue(RUNTIME_OK.endswith(":"))
        declared = dict(GENERATORS)
        try:
            GENERATORS["gen_moe.py"] = RUNTIME_OK + "#2043 - checks the fixture loads"
            self.assertEqual(contradicted({"gen_moe.py": "import onnxruntime\n"}), [])
            GENERATORS["gen_moe.py"] = "writes model fixtures, measures nothing"
            self.assertEqual(
                len(contradicted({"gen_moe.py": "import onnxruntime\n"})), 1
            )
        finally:
            GENERATORS.clear()
            GENERATORS.update(declared)

    def test_both_spellings_of_a_runtime_import_are_seen(self):
        # `from onnxruntime import InferenceSession` is the same behaviour as
        # `import onnxruntime`, and neither had a cell.
        self.assertTrue(looks_like_a_driver("from onnxruntime import InferenceSession\n"))
        self.assertTrue(looks_like_a_driver("import onnxruntime as ort\n"))

    def test_both_spellings_of_a_session_call_are_seen(self):
        # The attribute form is what the real CUDA driver uses.
        self.assertTrue(looks_like_a_driver("s = ort.InferenceSession(p)\n"))
        self.assertTrue(looks_like_a_driver("s = InferenceSession(p)\n"))

    def test_a_declared_driver_is_not_reported_as_contradicted(self):
        # Otherwise every driver would be reported twice, and the two findings
        # mean different things.
        self.assertEqual(contradicted({"ab.py": "import onnxruntime\n"}), [])


class Delegation(unittest.TestCase):
    """A `subprocess.run` one import away is still a benchmark starting."""

    def test_a_harness_that_calls_a_helper_is_a_harness(self):
        sources = {
            "lib.py": "import subprocess\n\ndef native(cmd):\n    subprocess.run(cmd)\n",
            "arm.py": "import lib as H\n\nH.native(cmd)\n",
        }
        table = spawning_helpers(sources)
        self.assertEqual(table["lib"], {"native"})
        self.assertTrue(looks_like_a_harness(sources["arm.py"], table))
        self.assertFalse(looks_like_a_harness(sources["arm.py"]))
        self.assertEqual(len(ep_failures(sources, {})), 2)

    def test_the_chain_resolves_further_than_one_level(self):
        # `acc0_gap_matrix.native` -> `acc0_w16_worker_split.collect` -> six
        # more files. Stopping at one level would clear all six.
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n",
            "mid.py": "import lib as H\n\ndef collect(c):\n    return H.native(c)\n",
            "top.py": "import mid as W\n\nW.collect(c)\n",
        }
        table = spawning_helpers(sources)
        self.assertEqual(table["mid"], {"collect"})
        self.assertTrue(looks_like_a_harness(sources["top.py"], table))

    def test_importing_the_harness_library_is_not_by_itself_running_one(self):
        # Nineteen files in that directory import `acc0_gap_matrix`; most use
        # its parsers. "Imports a harness" would have required a ledger entry
        # for every scorer in the tree, and a ledger nobody can maintain gets
        # a blanket exemption instead of a line.
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n\ndef parse(t):\n    return t.split()\n",
            "score.py": "import lib as H\n\nprint(H.parse(text))\n",
        }
        table = spawning_helpers(sources)
        self.assertFalse(looks_like_a_harness(sources["score.py"], table))
        self.assertEqual(ep_failures(sources, {"lib.py": "known-gap:#1 - x"}), [])

    def test_the_from_import_form_resolves_too(self):
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n",
            "arm.py": "from lib import native\n\nnative(c)\n",
            "quiet.py": "from lib import native\n",
        }
        table = spawning_helpers(sources)
        self.assertTrue(looks_like_a_harness(sources["arm.py"], table))
        self.assertFalse(looks_like_a_harness(sources["quiet.py"], table))

    def test_a_leaf_that_delegates_inside_its_own_file_still_counts(self):
        # The hole the first cut had, and the shape of the real tree:
        # `acc0_gap_matrix.native` contains no `subprocess` call at all, it
        # calls the same-file helper `sh`. Cross-module resolution alone left
        # `native` out of the table, so a file whose only call is `H.native`
        # read as starting nothing -- at width 16, four arms.
        sources = {
            "lib.py": "import subprocess\n\ndef sh(c):\n    subprocess.run(c, shell=True)\n\ndef native(b):\n    return sh(b)\n",
            "arm.py": "import lib as H\n\nH.native(b)\n",
        }
        table = spawning_helpers(sources)
        self.assertEqual(table["lib"], {"sh", "native"})
        self.assertTrue(looks_like_a_harness(sources["arm.py"], table))

    def test_the_real_helper_that_delegates_is_in_the_table(self):
        # Written against the table rather than the verdict, deliberately.
        # The cell below asserts six real files are seen as harnesses, and it
        # passed while `native` was missing -- because those six also call
        # `H.ort`, which spawns directly. A cell that names one mechanism and
        # is satisfied by another is how the ungated one stayed invisible.
        table = spawning_helpers(ep_sources())
        for helper in ("sh", "native", "ort", "competing_load"):
            self.assertIn(helper, table["acc0_gap_matrix"], helper)

    def test_the_harness_that_calls_only_native_is_recorded(self):
        # `acc0_w16_steal_ab.py`: four arms (fixed/steal1/steal4/null) at
        # width 16, entirely through `H.native`. Found by the fix above, not
        # by reading the directory.
        found = ep_sources()
        table = spawning_helpers(found)
        source = found["acc0_w16_steal_ab.py"]
        self.assertFalse(spawns_a_process(source))
        self.assertTrue(looks_like_a_harness(source, table))
        self.assertIn("acc0_w16_steal_ab.py", EP_LEDGER)

    def test_an_aliased_from_import_resolves_to_the_remote_name(self):
        # `from m import ort as run_ort`: the table holds `ort`, the call
        # site says `run_ort`. Looking the local name up in the remote
        # module's table finds nothing and clears the file -- fail-open.
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n",
            "arm.py": "from lib import native as go\n\ngo(cmd)\n",
        }
        self.assertTrue(
            looks_like_a_harness(sources["arm.py"], spawning_helpers(sources))
        )

    def test_a_star_import_does_not_hide_the_helper(self):
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n",
            "arm.py": "from lib import *\n\nnative(cmd)\n",
            "quiet.py": "from lib import *\n\nparse(text)\n",
        }
        table = spawning_helpers(sources)
        self.assertTrue(looks_like_a_harness(sources["arm.py"], table))
        self.assertFalse(looks_like_a_harness(sources["quiet.py"], table))

    def test_two_files_with_the_same_stem_are_unioned_not_overwritten(self):
        # An archived copy beside the live module: last-write-wins would
        # empty the live entry and clear every file importing it, from a file
        # nobody thought they were changing. Union is the fail-closed way.
        sources = {
            "lib.py": "import subprocess\n\ndef native(c):\n    subprocess.run(c)\n",
            "old/lib.py": "def native(c):\n    return 0\n",
            "arm.py": "import lib as H\n\nH.native(cmd)\n",
        }
        table = spawning_helpers(sources)
        self.assertEqual(table["lib"], {"native"})
        self.assertTrue(looks_like_a_harness(sources["arm.py"], table))

    def test_the_delegating_harnesses_in_the_tree_are_seen(self):
        # The real ones, and the reason this class exists: before the table,
        # `acc0_w16_study.py` and five others read as "starts nothing" while
        # running native and ORT arms at width 16 through `H.native`.
        found = ep_sources()
        table = spawning_helpers(found)
        for name in (
            "acc0_w16_study.py",
            "acc0_w8_w16_scaling.py",
            "acc0_w16_blocktime_ab.py",
            "acc0_w16_worker_split.py",
            "acc0_w16_chunk_permutation.py",
        ):
            self.assertFalse(spawns_a_process(found[name]), name)
            self.assertTrue(looks_like_a_harness(found[name], table), name)
        # Two of those five hold the lock, so the ledger must not list them.
        for gated in ("acc0_w16_worker_split.py", "acc0_w16_chunk_permutation.py"):
            self.assertTrue(holds_the_lock(found[gated]), gated)
            self.assertNotIn(gated, EP_LEDGER)


class EpBenches(unittest.TestCase):
    """The second root: crates/onnx-runtime-ep-cpu/benches."""

    def test_every_harness_there_holds_the_lock_or_carries_a_recorded_gap(self):
        self.assertEqual(ep_failures(ep_sources()), [])

    def test_the_directory_is_actually_found(self):
        # A path typo would make every cell above pass over an empty dict --
        # the same vacuity as a non-recursive glob, one root over.
        found = ep_sources()
        self.assertGreater(len(found), 20, "benches root not found or empty")
        self.assertIn("acc0_gap_matrix.py", found)

    def test_a_harness_one_directory_down_is_still_read(self):
        # The #2079 MUST FIX, transplanted: the workflow filter is
        # `crates/onnx-runtime-ep-cpu/benches/**`, so a push touching a file
        # in a subdirectory runs this job. A non-recursive glob would run it,
        # read nothing and go green. Planted rather than asserted, because a
        # pure-function assertion passes just as well with `glob` while the
        # directory has no subdirectory.
        planted = EP_BENCHES / "acc0_probe_tmp" / "sweep.py"
        planted.parent.mkdir(parents=True, exist_ok=True)
        try:
            planted.write_text("import subprocess\nsubprocess.run(cmd)\n")
            found = ep_sources()
            self.assertIn("acc0_probe_tmp/sweep.py", found)
            fail = ep_failures(found)
            self.assertEqual(len(fail), 1)
            self.assertIn("acc0_probe_tmp/sweep.py", fail[0])
        finally:
            planted.unlink(missing_ok=True)
            planted.parent.rmdir()

    def test_the_gated_harnesses_are_recognised_as_gated(self):
        # Not a restatement of the cell above: this pins that the *shell*
        # idiom counts, so a green run there means three files gate rather
        # than that the ledger grew three more lines.
        found = ep_sources()
        for name in (
            "acc0_gap_matrix.py",
            "acc0_w16_worker_split.py",
            "acc0_w16_chunk_permutation.py",
        ):
            self.assertTrue(holds_the_lock(found[name]), name)
            self.assertNotIn(name, EP_LEDGER)

    def test_a_new_ungated_harness_there_is_a_failure(self):
        fail = ep_failures({"acc0_brand_new.py": "import subprocess\nsubprocess.run(c)\n"})
        self.assertEqual(len(fail), 1)
        self.assertIn("carries no recorded gap", fail[0])

    def test_an_analysis_script_that_starts_nothing_needs_no_entry(self):
        # Half that directory reads JSON and prints tables. Requiring an
        # entry for those would make the ledger unmaintainable, and an
        # unmaintainable ledger is one that gets a blanket exemption.
        self.assertEqual(
            ep_failures({"acc0_w16_dispersion.py": "import json\njson.load(f)\n"}), []
        )
        self.assertFalse(looks_like_a_harness("import json\nprint(json.load(f))\n"))

    def test_an_unrecognised_reason_does_not_suppress_the_finding(self):
        source = "import subprocess\nsubprocess.run(c)\n"
        self.assertEqual(
            len(ep_failures({"x.py": source}, {"x.py": "known-gap:#2043 - later"})), 0
        )
        for reason in ("later", "", "wontfix", "gap:#2043"):
            fail = ep_failures({"x.py": source}, {"x.py": reason})
            self.assertEqual(len(fail), 1, reason)
            self.assertIn("unrecognised reason", fail[0])

    def test_a_process_that_is_not_a_benchmark_can_say_so(self):
        # Symmetry with `loads-runtime:` in the other root: a file shelling
        # out to `lscpu` should not be failed into pretending it saturates
        # the box, or the reason attached will be a lie by the second one.
        source = 'import subprocess\nsubprocess.run(["lscpu"])\n'
        self.assertEqual(
            ep_failures({"topo.py": source}, {"topo.py": EP_NOT_A_BENCH + "reads topology"}),
            [],
        )

    def test_every_recorded_gap_names_the_audit_issue(self):
        # A gap with no issue attached is a gap nobody is going to close.
        for name, reason in EP_LEDGER.items():
            if not reason.startswith(EP_GAP):
                continue
            self.assertTrue(reason.startswith(EP_GAP + "#"), name)

    def test_a_wrapper_is_recorded_as_a_wrapper_not_as_a_gap(self):
        # `acc0_nothp_exec.py` execs the command it is handed, so the lock
        # belongs to its caller. Taking it here would be worse than not: exec
        # keeps the pid and never runs the release, so the lock would outlive
        # the run and have to be reaped.
        self.assertTrue(EP_LEDGER["acc0_nothp_exec.py"].startswith(EP_WRAPPER))
        self.assertFalse(holds_the_lock(ep_sources()["acc0_nothp_exec.py"]))


class EpShellHarnesses(unittest.TestCase):
    """The same root, read in the other language it is written in.

    Every cell here would have passed vacuously before the enumeration
    existed, because the glob that fed it returned nothing. The live positive
    control below is what makes a green run mean something.
    """

    def test_every_shell_harness_there_gates_or_says_why(self):
        self.assertEqual(ep_shell_failures(ep_shell_sources()), [])

    def test_the_shell_files_are_actually_found(self):
        # The vacuity that would make this whole class free: a `*.sh` glob
        # over a directory it cannot see returns `{}`, and `ep_shell_failures`
        # is happiest with nothing to check.
        found = ep_shell_sources()
        self.assertGreater(len(found), 0, "benches root has no shell files")
        self.assertIn("decode_placement_census.sh", found)

    def test_the_census_is_read_as_gated(self):
        # The live positive control, and the file that motivated the check:
        # its header claimed the lock for weeks while the script never called
        # `hostlock.sh`. If a future edit drops the re-exec, this fails by
        # name rather than the ledger quietly growing a line.
        source = ep_shell_sources()["decode_placement_census.sh"]
        self.assertTrue(shell_holds_the_lock(source))
        self.assertNotIn("decode_placement_census.sh", EP_SHELL_LEDGER)

    def test_the_census_takes_it_once_around_every_arm(self):
        # #1803's property as far as reading the file can pin it: exactly one
        # *acquiring* invocation, and it precedes the definition of `run()`
        # and all of its calls. A lock taken per arm is released between
        # them, which is the window a peer sampled the host in and read it as
        # clear.
        #
        # Positional, and deliberately not sold as more: what actually holds
        # the lock across all three arms is the whole-script re-exec, and the
        # run-time half of the same question is the `host_lock` column on an
        # emitted row. `taken` counts acquisitions rather than mentions, so
        # switching the census to `hostlock.sh status` fails here as well as
        # in the cell above.
        lines = strip_shell_comments(
            ep_shell_sources()["decode_placement_census.sh"]
        ).splitlines()
        taken = [i for i, ln in enumerate(lines) if shell_holds_the_lock(ln)]
        self.assertEqual(len(taken), 1, taken)
        defined = [i for i, ln in enumerate(lines) if re.match(r"\s*run\s*\(\)", ln)]
        called = [i for i, ln in enumerate(lines) if re.match(r'\s*run\s+"', ln)]
        self.assertEqual(len(defined), 1, defined)
        self.assertGreaterEqual(len(called), 3, called)
        self.assertLess(taken[0], min(defined + called))

    def test_the_census_decides_by_custody_not_by_an_inherited_variable(self):
        # Review of this file found the re-exec guarded by an exported
        # sentinel. A sentinel is an ordinary inheritable variable: any
        # unrelated ancestor exporting that name sends the census through
        # three saturating pool launches with **no lock at all**, silently --
        # #1803's hazard wearing the costume of the fix for it.
        #
        # So the guard reads custody instead: the holder pid from the lock
        # itself, walked against this process's own ancestry. Asserted here
        # by mechanism rather than by outcome, because the outcome (it ran
        # once, it acquired once) is identical either way -- which is exactly
        # why the defect survived a working end-to-end run.
        source = ep_shell_sources()["decode_placement_census.sh"]
        code = strip_shell_comments(source)
        self.assertIn("hostlock.sh status", code)
        self.assertIn("/proc/$1/stat", code)
        self.assertRegex(code, r"\bHELD by\b")
        self.assertNotIn("HOSTLOCK_CENSUS_HELD", code)
        # And the ancestry walk has to be a walk: a single `$PPID` compare
        # is wrong the moment the lock is held by a grandparent, which is
        # the normal case under `hostlock.sh run` inside another harness.
        self.assertRegex(code, r"while \[.*\$_p.*\]")

    def test_the_census_parses_the_status_line_the_lock_actually_prints(self):
        # Naming `HELD by` is not parsing it. This runs the census's own
        # `sed` -- lifted out of the file, not restated -- over a line built
        # from `hostlock.sh`'s own `echo`, so a format change on either side
        # is a failure here rather than a census that silently degrades to
        # "always re-acquire" and quietly stops nesting.
        census = ep_shell_sources()["decode_placement_census.sh"]
        script = re.search(r"sed -n '([^']*HELD by[^']*)'", census)
        self.assertIsNotNone(script, "the status parse moved or changed shape")

        lock = (ORT_AB.parents[1] / "scripts" / "hostlock.sh").read_text()
        template = re.search(r'echo "(HELD by \$\{[^"]*)"', lock)
        self.assertIsNotNone(template, "hostlock.sh's HELD line moved")
        line = (
            template.group(1)
            .replace("${owner}", "justinchu")
            .replace("${pid}", "31337")
            .replace("${age}", "12")
            .replace("${at}", "2026-08-25T20:46:18Z")
        )
        self.assertNotIn("${", line, line)

        def parsed(text: str) -> str:
            return subprocess.run(
                ["sed", "-n", script.group(1)],
                input=text,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        self.assertEqual(parsed(line + "\n  reason: x\n"), "31337")
        # And the states that are not custody yield nothing, so the census
        # re-execs rather than reading somebody's expired claim as its own.
        for other in ("FREE  (runnable=3)", "STALE by u pid=1 (holder gone)"):
            self.assertEqual(parsed(other + "\n"), "")

    def test_a_new_ungated_shell_harness_there_is_a_failure(self):
        fail = ep_shell_failures({"knee.sh": "#!/bin/sh\ncargo bench --bench x\n"})
        self.assertEqual(len(fail), 1)
        self.assertIn("carries no recorded reason", fail[0])

    def test_a_shell_file_one_directory_down_is_still_read(self):
        # Planted rather than asserted, for the reason the Python cell gives:
        # a non-recursive glob passes a pure-function test just as well while
        # the directory has no subdirectory, and the workflow filter is `**`.
        planted = EP_BENCHES / "acc0_shell_tmp" / "sweep.sh"
        planted.parent.mkdir(parents=True, exist_ok=True)
        try:
            planted.write_text("#!/bin/sh\ncargo bench --bench x\n")
            found = ep_shell_sources()
            self.assertIn("acc0_shell_tmp/sweep.sh", found)
            fail = ep_shell_failures(found)
            self.assertEqual(len(fail), 1)
            self.assertIn("acc0_shell_tmp/sweep.sh", fail[0])
        finally:
            planted.unlink(missing_ok=True)
            # Suppressed: if the body raised, `rmdir` raising here would
            # replace the real failure with a directory-not-empty error and
            # the diagnosis would start in the wrong place. An empty stray
            # directory is inert -- `ep_shell_sources` globs `*.sh`.
            with contextlib.suppress(OSError):
                planted.parent.rmdir()

    def test_naming_the_script_is_not_holding_the_lock(self):
        # The #2106 defect, in the other direction: a substring check for
        # `hostlock.sh` counts every one of these as custody, and each of
        # them is a script reading or waiting on the host without claiming it.
        for sub in ("status", "provenance", "wait", "release", "status --porcelain"):
            self.assertFalse(
                shell_holds_the_lock(f"./scripts/hostlock.sh {sub}\n"), sub
            )
        self.assertFalse(shell_holds_the_lock("echo see scripts/hostlock.sh\n"))

    def test_both_acquiring_subcommands_count(self):
        for sub in SHELL_ACQUIRING:
            self.assertTrue(shell_holds_the_lock(f"scripts/hostlock.sh {sub} --wait\n"))

    def test_a_commented_out_acquisition_does_not_count(self):
        # The exact shape of the defect this class was written for: the claim
        # lived in a comment and the call did not exist.
        #
        # The last three are the shapes the first version of the stripper got
        # **wrong**, each of them fail-open -- a real comment read as custody,
        # which under a categorical rule is a silent exemption with no ledger
        # line. They are here rather than in a note because a documented
        # weakness that no cell exercises is a weakness that comes back.
        for line in (
            "# it runs under scripts/hostlock.sh run as a courtesy\n",
            "  # ./scripts/hostlock.sh acquire --wait\n",
            "echo hi  # scripts/hostlock.sh run -- x\n",
            "echo hi;# scripts/hostlock.sh run -- x\n",
            "echo \"it's fine\"  # scripts/hostlock.sh run -- x\n",
            "echo '\"'  # scripts/hostlock.sh run -- x\n",
            # A comment containing a separator ahead of the mention. Found by
            # mutation: with the word-start rule weakened to whitespace only,
            # the `;` inside the comment reads as command position and the
            # line certifies the file.
            "echo hi;# see; ./scripts/hostlock.sh run -- x\n",
        ):
            self.assertFalse(shell_holds_the_lock(line), line)

    def test_printing_the_recipe_is_not_running_it(self):
        # Usage text telling the reader to run under the lock is the single
        # most plausible false positive in a bench directory, and it is the
        # census's own defect one syntactic layer up: a declaration with
        # nothing behind it.
        #
        # The last three came from review of this file, and each was a live
        # fail-open: a multi-line quoted argument whose interior newline put
        # the following *arguments* at the start of a line, where they read
        # as a command; and a heredoc ended early by an indented
        # delimiter-lookalike, which real `sh` does not treat as a
        # terminator (`<<-` strips tabs, and nothing strips spaces).
        for source in (
            'usage(){ echo "run under: scripts/hostlock.sh run -- $0"; }\n',
            'echo see scripts/hostlock.sh run\n',
            'printf "%s\\n" "scripts/hostlock.sh run -- x"\n',
            'cat <<EOF\nrun it under scripts/hostlock.sh run -- x\nEOF\n',
            "cat <<'EOF'\nscripts/hostlock.sh run -- x\nEOF\n",
            "cat <<-EOT\n\tscripts/hostlock.sh acquire\n\tEOT\n",
            'echo "\nscripts/hostlock.sh run -- x"\n',
            "printf 'intro\\ninvoke: %s %s\\n' scripts/hostlock.sh run\n",
            'echo "a\nb" ./scripts/hostlock.sh run\n',
            "cat <<EOF\n  EOF\n./scripts/hostlock.sh run -- x\nEOF\n",
            # Delimiters that are not `\w+`. `<<'EO-F'` captured `EO`, whose
            # terminator never arrives, and the body was then read as code --
            # an undocumented silent exemption, and `<<'END-OF-USAGE'` is an
            # ordinary thing to write.
            "cat <<'EO-F'\nscripts/hostlock.sh run -- x\nEO-F\n",
            "cat <<END.TXT\nscripts/hostlock.sh run -- x\nEND.TXT\n",
            # Enough whitespace to defeat a fixed-width lookback in front of
            # the quoted delimiter, which is why that search runs from the
            # start of the line.
            "cat <<       'EOF'\nscripts/hostlock.sh run -- x\nEOF\n",
            # A heredoc whose delimiter never arrives. `sh` feeds the rest of
            # the file to `cat` and the script exits 0 -- it runs, and it
            # holds nothing.
            "cat <<EOF\nscripts/hostlock.sh run -- x\n",
            "cat <<EOF\r\nscripts/hostlock.sh run -- x\r\nEOF\r\n",
        ):
            self.assertFalse(shell_holds_the_lock(source), source)

    def test_a_run_below_something_that_is_not_a_heredoc_still_counts(self):
        # The other direction of the same machinery, and the one that costs
        # somebody else a false ledger line in their lane -- which is how a
        # conformance check gets deleted rather than obeyed.
        #
        # Each of these has a `<<` that is not a heredoc operator, or a quote
        # that is not what it looks like, followed by a real acquisition. The
        # first version blanked everything below the `<<` to end of file and
        # reported all four as ungated. `hostlock.sh` itself uses `<<<`.
        for source in (
            "n=$(( 1 << k ))\n./scripts/hostlock.sh run -- x\n",
            "# shift << bits\n./scripts/hostlock.sh run -- x\n",
            'echo "x << 2"\n./scripts/hostlock.sh run -- x\n',
            "cat <<<word\n./scripts/hostlock.sh run -- x\n",
            "msg=$'can\\'t'\n./scripts/hostlock.sh run -- x\n",
            "cat <<EOF\nhello\nEOF\n./scripts/hostlock.sh run -- x\n",
            # A heredoc that *does* end has to give the rest of the file
            # back, and these are the two shapes where a narrower terminator
            # rule silently does not: a bare delimiter that is not `\w+`
            # (`\w+` captured `END`, which never arrives), and CRLF line
            # endings (the terminator line is `EOF\r`). Both fail closed, so
            # only a positive case can see them -- the false-cases above
            # pass either way, which is exactly why they were not enough.
            "cat <<END.TXT\nhi\nEND.TXT\n./scripts/hostlock.sh run -- x\n",
            "cat <<EOF\r\nhi\r\nEOF\r\n./scripts/hostlock.sh run -- x\r\n",
            # The quoted delimiter, positively. Every other cell for
            # `<<'EOF'` is a false-case, and review showed they all still
            # pass with the quoted handling removed entirely -- because the
            # bare path then captures `'EO-F'` *with* its quotes, which also
            # never terminates, and over-blanking hides an over-blanking bug.
            # Only a real acquisition below the terminator can tell the two
            # apart.
            "cat <<'EOF'\nhi\nEOF\n./scripts/hostlock.sh run -- x\n",
            # Delimiters the shell quote-removes but that are neither wholly
            # quoted nor wholly bare. `<<\\EOF` is an ordinary way to write
            # `<<'EOF'`; matching it as `\\EOF` left the terminator unfound.
            "cat <<\\EOF\nhi\nEOF\n./scripts/hostlock.sh run -- x\n",
            'cat <<E"O"F\nhi\nEOF\n./scripts/hostlock.sh run -- x\n',
            # The delimiter word ends at a shell metacharacter, so what
            # follows the operator on the same line is not part of it. Found
            # by mutation: widening the bare class to `[^\s]` captured
            # `EOF;` and `EOF>out`, neither of which ever terminates, and
            # the acquisition below was blanked with the rest of the file.
            "cat <<EOF; echo hi\nbody\nEOF\n./scripts/hostlock.sh run -- x\n",
            "(cat <<EOF)\nbody\nEOF\n./scripts/hostlock.sh run -- x\n",
            # The last two discriminate the two *narrower* guards from the
            # broad one above them: excluding `<<<`, and running the heredoc
            # scan over a copy in which quoted regions are blanked. Both are
            # defence in depth -- with only the "a delimiter that never
            # arrives blanks nothing" rule, a phantom heredoc is harmless
            # until some later line happens to equal its delimiter, which is
            # what these two supply. Contrived inputs, deliberately: the
            # mechanism is what is being pinned, and a guard no cell can
            # distinguish is a guard that gets deleted as dead weight.
            "cat <<<word\n./scripts/hostlock.sh run -- x\nword\n",
            'echo "x << 2"\n./scripts/hostlock.sh run -- x\n2\n',
            # Arithmetic is not a redirection. Both forms, and with the
            # shift's right-hand word later alone on a line, which is what
            # makes the exclusion load-bearing rather than incidental: an
            # unterminated heredoc consumes the rest of the file.
            "(( n = 1 << k ))\n./scripts/hostlock.sh run -- x\n",
            "n=$(( 1 << k ))\n./scripts/hostlock.sh run -- x\nk\n",
            # A `((` spanning lines is two subshells, not arithmetic, and a
            # heredoc opened inside it still has to be seen.
            "( (cat <<EOF\nhi\nEOF\n) )\n./scripts/hostlock.sh run -- x\n",
        ):
            self.assertTrue(shell_holds_the_lock(source), source)

    def test_naming_the_path_is_not_running_it(self):
        # `LOCK=scripts/hostlock.sh` assigns a path and runs nothing. The
        # subcommand is read from the same logical line for exactly this
        # reason: scanning to the end of the file let an unrelated `run` two
        # lines down certify the file.
        self.assertFalse(shell_holds_the_lock("LOCK=scripts/hostlock.sh\nrun --foo\n"))
        self.assertFalse(shell_holds_the_lock("./my-hostlock.sh run -- x\n"))
        # A differently-named wrapper *in* an assignment. Found by mutation:
        # command position alone accepts this one, because `LOCK=./my-` parses
        # as an assignment and clears the head -- so the left boundary on the
        # match is load-bearing here rather than decorative.
        self.assertFalse(shell_holds_the_lock("LOCK=./my-hostlock.sh run -- x\n"))

    def test_a_quoted_path_or_a_continuation_still_counts(self):
        # The other direction, and the reason single-line quoted regions are
        # kept rather than blanked: a quoted word is very often the command.
        for source in (
            '"$REPO/scripts/hostlock.sh" run --reason x -- ./bench\n',
            'exec "$ROOT/scripts/hostlock.sh" run --wait -- "$@"\n',
            "./scripts/hostlock.sh \\\n  run \\\n  --wait -- ./bench\n",
            "exec ./scripts/hostlock.sh 'run' --wait -- \"$SELF\"\n",
            "TMPDIR=/x env ./scripts/hostlock.sh acquire --wait\n",
            "if true; then ./scripts/hostlock.sh run -- x; fi\n",
            "for a in 1 2; do ./scripts/hostlock.sh acquire; done\n",
            "cmd && ./scripts/hostlock.sh run -- x\n",
            # Every separator in the command-position set, because a mutant
            # deleting `|`, `(`, `{` or a backtick from it survived the
            # battery: nothing else here reaches those branches.
            "foo | ./scripts/hostlock.sh run -- x\n",
            "( ./scripts/hostlock.sh run -- x )\n",
            "{ ./scripts/hostlock.sh acquire; }\n",
            "v=`./scripts/hostlock.sh run -- x`\n",
            "case $x in a) ./scripts/hostlock.sh run -- y ;; esac\n",
        ):
            self.assertTrue(shell_holds_the_lock(source), source)

    def test_comment_stripping_keeps_the_shell_that_needs_a_hash(self):
        # `${x#-}` and `$#` are not comments. Truncating there would drop a
        # real acquisition below them and report a gated file as ungated --
        # a false alarm in somebody else's lane, which is how a check gets
        # deleted rather than obeyed.
        source = 'n=$#\narg=${1#-}\n./scripts/hostlock.sh run -- ./bench\n'
        self.assertIn("${1#-}", strip_shell_comments(source))
        self.assertTrue(shell_holds_the_lock(source))
        quoted = 'echo "# not a comment"\nscripts/hostlock.sh run -- x\n'
        self.assertTrue(shell_holds_the_lock(quoted))

    def test_an_unrecognised_reason_does_not_suppress_the_finding(self):
        source = "#!/bin/sh\ncargo bench --bench x\n"
        self.assertEqual(
            len(ep_shell_failures({"x.sh": source}, {"x.sh": "known-gap:#2043 - later"})),
            0,
        )
        for reason in ("later", "", "wontfix", "gap:#2043"):
            fail = ep_shell_failures({"x.sh": source}, {"x.sh": reason})
            self.assertEqual(len(fail), 1, reason)
            self.assertIn("unrecognised reason", fail[0])

    def test_a_wrapper_or_a_non_benchmark_can_say_so(self):
        source = "#!/bin/sh\nexec \"$@\"\n"
        self.assertEqual(
            ep_shell_failures({"w.sh": source}, {"w.sh": "wrapper:#1 - caller gates"}),
            [],
        )
        self.assertEqual(
            ep_shell_failures({"t.sh": source}, {"t.sh": "no-bench:#1 - reads lscpu"}),
            [],
        )

    def test_the_readme_says_what_is_true_of_these_two_files(self):
        # The doc names both files and states opposite things about them:
        # one gates, one is recorded. Asserting only that both names appear
        # would pass with the two descriptions **swapped**, which is the doc
        # drifting into fiction -- the failure this class exists for. So the
        # claim is read out of the sentence each name sits in.
        doc = (ORT_AB / "README.md").read_text()
        sources = ep_shell_sources()

        def sentence(name: str) -> str:
            at = doc.index(name)
            start = max(doc.rfind(". ", 0, at), doc.rfind("\n\n", 0, at)) + 1
            ends = [e for e in (doc.find(". ", at), doc.find("\n\n", at)) if e > 0]
            text = doc[start : min(ends) if ends else len(doc)].lower()
            # Emphasis and line wrapping are formatting, not claims: `**takes
            # the lock**` split across two lines is the same sentence.
            return " ".join(text.replace("*", "").replace("`", "").split())

        census = sentence("decode_placement_census.sh")
        # "takes the lock", not the bare token `lock` -- which "hostlock.sh"
        # contains, so the weaker assertion held even for a sentence saying
        # the opposite. Substring matching still cannot see a negation, so
        # the ones a rewrite would plausibly reach for are named.
        self.assertIn("takes the lock", census)
        for negation in ("no longer", "does not", "used to", "should", "never"):
            self.assertNotIn(negation, census, census)
        self.assertNotIn("known-gap", census)
        self.assertTrue(shell_holds_the_lock(sources["decode_placement_census.sh"]))

        modulo = sentence("int4_modulo_arms.sh")
        self.assertIn("known-gap", modulo)
        # The mirror of the assertion above. It does not discriminate against
        # today's prose -- the paragraph after this sentence happens not to
        # say "takes the lock" -- so it is a drift guard for a rewrite that
        # moves the two descriptions adjacent, not a control that proves the
        # boundary is tight.
        self.assertNotIn("takes the lock", modulo)
        self.assertTrue(EP_SHELL_LEDGER["int4_modulo_arms.sh"].startswith(EP_GAP))
        self.assertFalse(shell_holds_the_lock(sources["int4_modulo_arms.sh"]))

    def test_the_readme_does_not_still_describe_a_rule_that_was_replaced(self):
        # This file's own recurring defect, applied to its own prose: the
        # README described an unterminated heredoc as blanking nothing long
        # after that had become the opposite -- and it was fail-open besides,
        # since a body read as code is exactly the false custody the pass
        # exists to refuse. So the behaviour is asserted here, and the doc is
        # required to state the rule that actually holds.
        #
        # Stated positively rather than as a banned phrase: the retraction a
        # few lines further down the README quotes the old wording verbatim,
        # and a check that cannot tell a claim from a correction of it would
        # have to be satisfied by deleting the correction.
        unterminated = "cat <<EOF\n./scripts/hostlock.sh run -- x\n"
        self.assertNotIn("hostlock", strip_shell_comments(unterminated))
        self.assertFalse(shell_holds_the_lock(unterminated))
        doc = " ".join((ORT_AB / "README.md").read_text().split())
        self.assertIn("a delimiter that never arrives consumes the rest of", doc)

    def test_the_shell_ledger_is_checked_in_both_directions(self):
        self.assertEqual(dead_shell_records(ep_shell_sources()), [])
        self.assertEqual(stale_shell_records(ep_shell_sources()), [])
        self.assertEqual(
            dead_shell_records({}, {"gone.sh": "known-gap:#1 - x"}),
            ["gone.sh: recorded, but no such file -- delete the record"],
        )
        gated = {"g.sh": "scripts/hostlock.sh run -- x\n"}
        fail = stale_shell_records(gated, {"g.sh": "known-gap:#1 - x"})
        self.assertEqual(len(fail), 1)
        self.assertIn("delete the record", fail[0])


class EpRustBenches(unittest.TestCase):
    """The third language, pinned rather than gated, while #2129 is open."""

    def test_the_rust_bench_set_is_the_one_that_was_deferred(self):
        # The whole value is in the failure: this cell exists so that adding
        # a fourteenth `[[bench]]` cannot happen silently while none of the
        # thirteen takes the lock.
        found = ep_rust_benches()
        self.assertEqual(
            found,
            EP_RUST_UNCOVERED,
            "the EP crate's [[bench]] set moved -- gate the new target's "
            "`cargo bench` invocation under scripts/hostlock.sh, or add it "
            "here and say why on #2129",
        )

    def test_the_pin_is_not_vacuous(self):
        # Two ways this could pass while reading nothing: an empty manifest
        # parse compared against an empty constant, and a parser that finds
        # `name` keys outside `[[bench]]`. So: a live non-empty assertion,
        # the two targets named in the incident reports, and a negative --
        # the crate's `[package] name` must not leak in.
        found = ep_rust_benches()
        self.assertGreaterEqual(len(found), 13)
        self.assertIn("decode_gap_park_ab", found)
        self.assertIn("native_vs_mlas", found)
        self.assertNotIn("onnx-runtime-ep-cpu", found)

    def test_every_pinned_bench_has_a_source_file(self):
        # A stale pin is the shell ledger's dead record in another form: a
        # name kept here after its target was deleted still passes the set
        # comparison only if the manifest kept it too, but a `#[bench]` whose
        # file is gone is a manifest that will not build. Cheap to state.
        for name in sorted(ep_rust_benches()):
            self.assertTrue(
                (EP_BENCHES / f"{name}.rs").exists(), f"{name}.rs is missing"
            )


class Staleness(unittest.TestCase):
    """A record that has come true is a record that must go."""

    def test_no_recorded_gap_in_the_tree_is_stale(self):
        self.assertEqual(stale_records(ep_sources(), EP_LEDGER), [])
        self.assertEqual(stale_records(read_sources(), DRIVERS), [])

    def test_a_gap_that_has_since_been_gated_fails(self):
        # The drift this prevents: somebody gates `acc0_w16_page_backing.py`,
        # the ledger still exempts it, and if the gate is later removed the
        # stale line silently covers it again. Closing a gap is a one-line
        # edit here, and the suite says so by name.
        gated = {"acc0_w16_page_backing.py": "with H.HostLock(owner='roy'):\n    pass\n"}
        fail = stale_records(gated, EP_LEDGER)
        self.assertEqual(len(fail), 1)
        self.assertIn("delete the record", fail[0])

    def test_the_same_rule_covers_the_other_root(self):
        cuda = {"ort_cuda_decode_bench.py": "hostlock_gate.require(cmd)\n"}
        self.assertEqual(len(stale_records(cuda, DRIVERS)), 1)
        self.assertEqual(stale_records({"ab.py": "hostlock_gate.require(c)\n"}, DRIVERS), [])

    def test_no_record_points_at_a_file_that_is_gone(self):
        self.assertEqual(dead_records(ep_sources()), [])

    def test_a_record_for_a_deleted_or_defanged_file_fails(self):
        self.assertEqual(
            dead_records({}, {"gone.py": "known-gap:#2043 - x"}),
            ["gone.py: recorded, but no such file -- delete the record"],
        )
        quiet = {"gone.py": "import json\n"}
        self.assertIn("no longer starts anything", dead_records(quiet, {"gone.py": "known-gap:#1"})[0])


class SpawnDetection(unittest.TestCase):
    def test_the_common_spellings_all_count(self):
        for source in (
            "import subprocess\nsubprocess.run(cmd)\n",
            "import subprocess\nsubprocess.Popen(cmd)\n",
            "import subprocess\nsubprocess.check_output(cmd)\n",
            "import os\nos.system(cmd)\n",
            "from subprocess import run\nrun(cmd)\n",
            "from subprocess import Popen as P\nP(cmd)\n",
            "import subprocess as sp\nsp.run(cmd)\n",
        ):
            self.assertTrue(spawns_a_process(source), source)

    def test_the_process_starting_idioms_all_count(self):
        # None of these is in the tree today; a benchmark directory is
        # exactly where the first one will appear, and the cost of the
        # omission is a saturating harness that reads as harmless.
        for source in (
            "import os\nos.fork()\n",
            "import multiprocessing\nmultiprocessing.Process(target=f).start()\n",
            "from multiprocessing import Pool\nPool(8)\n",
            "from concurrent.futures import ProcessPoolExecutor\nProcessPoolExecutor(8)\n",
            "import concurrent.futures\nconcurrent.futures.ProcessPoolExecutor(8)\n",
        ):
            self.assertTrue(spawns_a_process(source), source)

    def test_a_dotted_module_path_is_read_whole(self):
        # `import concurrent.futures` binds `concurrent`, so the call site is
        # two attributes deep. A one-step `Name.attr` match reads it as
        # something else entirely and clears it.
        self.assertEqual(
            dotted_name(ast.parse("a.b.c(1)").body[0].value.func), "a.b.c"
        )
        self.assertIsNone(dotted_name(ast.parse("f()(1)").body[0].value.func))

    def test_something_else_called_run_is_not_a_spawn(self):
        # `pool.run(...)`, `bench.run(...)`: an attribute call on anything
        # that is not the subprocess module.
        self.assertFalse(spawns_a_process("import subprocess\npool.run(cmd)\n"))
        self.assertFalse(spawns_a_process("def run(c):\n    pass\n\nrun(1)\n"))

    def test_prose_about_running_a_benchmark_is_not_a_spawn(self):
        self.assertFalse(spawns_a_process('"""Run subprocess.run(cmd) yourself."""\n'))
        self.assertFalse(spawns_a_process("# subprocess.run(cmd)\n"))

    def test_the_real_harnesses_are_detected(self):
        # Anti-vacuity for the ledger: if the detector regressed to False the
        # ledger cells above would all pass, having found nothing to check.
        found = ep_sources()
        table = spawning_helpers(found)
        detected = sorted(n for n, s in found.items() if looks_like_a_harness(s, table))
        # Floors and ceilings, both load-bearing: a detector stuck on False
        # would let every ledger cell above pass having found nothing, and one
        # stuck on True would "detect" the analysis scripts and make the
        # ledger unmaintainable. Keyed off the record rather than a literal.
        self.assertGreaterEqual(len(detected), len(EP_LEDGER))
        self.assertLess(len(detected), len(found))
        for name in EP_LEDGER:
            self.assertIn(name, detected)
        for quiet in ("acc0_w16_dispersion.py", "acc0_w16_mode_stratified.py"):
            self.assertNotIn(quiet, detected)


class LockIdioms(unittest.TestCase):
    def test_the_shell_lock_counts_in_both_import_styles(self):
        self.assertTrue(uses_the_shell_lock("with H.HostLock(owner='roy'):\n    pass\n"))
        self.assertTrue(
            uses_the_shell_lock(
                "from acc0_gap_matrix import HostLock\nwith HostLock():\n    pass\n"
            )
        )

    def test_naming_the_class_without_using_it_is_not_holding_it(self):
        # Fail-closed direction, same as the python gate: an import or a
        # mention is not an acquisition.
        for source in (
            "import acc0_gap_matrix as H\n",
            "# with H.HostLock(): ...\n",
            '"""Wrap the sweep in H.HostLock."""\n',
        ):
            self.assertFalse(uses_the_shell_lock(source), source)

    def test_reading_the_lock_state_is_not_holding_it(self):
        # Several harnesses print `hostlock_state` out of a saved JSON blob.
        # Reporting a lock somebody else held is exactly the label-without-
        # custody failure the column was added to prevent.
        source = 'state = blob.get("hostlock", {}).get("hostlock_state")\n'
        self.assertFalse(holds_the_lock(source))
        self.assertEqual(len(ep_failures({"x.py": "import subprocess\nsubprocess.run(c)\n" + source})), 1)

    def test_either_idiom_satisfies_the_benches_root(self):
        for source in (
            "import subprocess\nsubprocess.run(c)\nwith H.HostLock():\n    pass\n",
            "import subprocess\nsubprocess.run(c)\nhostlock_gate.require(c)\n",
        ):
            self.assertEqual(ep_failures({"x.py": source}), [], source)


class Custody(unittest.TestCase):
    """The holder is the outer harness, not each arm."""

    def test_nothing_in_the_tree_takes_the_lock_inside_a_loop(self):
        for sources in (read_sources(), ep_sources()):
            for name, source in sources.items():
                if name in LIBRARIES or name in TESTS:
                    continue
                self.assertEqual(acquired_inside_a_loop(source), [], name)

    def test_a_lock_taken_per_arm_is_caught(self):
        # #1803's mechanism, in source form: acquiring inside the arm loop
        # releases between arms, so the A and the B are separately protected
        # and the comparison is not protected at all.
        per_arm = "for arm in arms:\n    with H.HostLock():\n        run(arm)\n"
        self.assertEqual(acquired_inside_a_loop(per_arm), [2])
        outer = "with H.HostLock():\n    for arm in arms:\n        run(arm)\n"
        self.assertEqual(acquired_inside_a_loop(outer), [])

    def test_the_python_gate_is_held_the_same_way(self):
        per_arm = "while more:\n    hostlock_gate.require(cmd)\n"
        self.assertEqual(acquired_inside_a_loop(per_arm), [2])

    def test_the_loop_is_found_inside_a_function_too(self):
        # Where it actually lives: no harness in the tree loops at module
        # level, they all loop inside `main()`. A check that read only
        # top-level statements would pass over every real file.
        in_main = (
            "def main():\n"
            "    for arm in arms:\n"
            "        with H.HostLock():\n"
            "            run(arm)\n"
        )
        self.assertEqual(acquired_inside_a_loop(in_main), [3])
        outer = (
            "def main():\n"
            "    with H.HostLock():\n"
            "        for arm in arms:\n"
            "            run(arm)\n"
        )
        self.assertEqual(acquired_inside_a_loop(outer), [])

    def test_a_nested_loop_does_not_hide_the_acquisition(self):
        deep = (
            "for a in arms:\n"
            "    for r in reps:\n"
            "        with HostLock():\n"
            "            run(a)\n"
        )
        self.assertEqual(acquired_inside_a_loop(deep), [3])



class GateAdmission(unittest.TestCase):
    """A quality gate must report *who* it discarded, not just how many.

    The rest of this file is static analysis. These are behavioural, because
    the property is about accounting a reader cannot see by reading the source
    -- whether the counter that lands in the artifact is attributed to the arm
    that was actually rejected.

    Why it matters here: `int4_modulo_matrix.py` discards any launch whose
    rusage CPU efficiency falls below a floor, and the arms it compares have
    genuinely different runtimes. A fixed floor can therefore admit them at
    different rates and silently select which launches of an arm survive. The
    harness previously kept one aggregate counter, which cannot distinguish
    that from an even-handed gate.

    The import is inside each test so a breakage in the bench module fails
    these tests rather than the collection of this whole file.
    """

    @staticmethod
    def _harness():
        if str(EP_BENCHES) not in sys.path:
            sys.path.insert(0, str(EP_BENCHES))
        import int4_modulo_matrix

        return int4_modulo_matrix

    def test_the_rate_is_per_arm_and_an_empty_run_does_not_divide_by_zero(self):
        m = self._harness()
        cols = m.admission_columns({"before": 3, "after": 0}, {"before": 12, "after": 12})
        self.assertEqual(cols["total"], 3)
        self.assertEqual(cols["by_arm"], {"before": 3, "after": 0})
        self.assertAlmostEqual(cols["rate_by_arm"]["before"], 0.25)
        self.assertAlmostEqual(cols["rate_spread"], 0.25)
        # A skipped matrix reports zeroes rather than raising on 0/0.
        self.assertEqual(m.admission_columns({}, {})["rate_spread"], 0.0)

    def test_a_discard_is_attributed_to_the_arm_that_was_actually_rejected(self):
        """The assertion an aggregate counter cannot make.

        Only `before` is starved. A harness that counted a total, or that
        attributed to the wrong arm, still reports three discards -- and only
        the per-arm breakdown says the gate was one-sided, which is exactly
        the case where the surviving launches of that arm are a biased sample.
        """
        m = self._harness()
        rounds, starved, seen = 4, "before", []

        def fake_launch(binary, env_extra, timeout=1800):
            arm = Path(binary).name.replace("prefill_", "")
            seen.append(arm)
            # Starve one arm on the first round only, so the arm still has
            # surviving samples and the matrix reaches its table.
            first_touch = seen.count(arm) == 1
            eff = 0.10 if (arm == starved and first_touch) else 0.99
            rows = {1: {"steady_ms": 1.0, "cold_ms": 2.0, "fnv": "abcd"}}
            return rows, eff

        with unittest.mock.patch.object(m, "launch", fake_launch):
            _table, adm = m.prefill_matrix(rounds, 16, "shape", [1])

        self.assertEqual(adm["by_arm"][starved], 1, adm)
        self.assertEqual(adm["attempts_by_arm"][starved], rounds, adm)
        for arm in ("after", "aa"):
            self.assertEqual(adm["by_arm"][arm], 0, adm)
        self.assertEqual(adm["total"], 1)
        # Only one arm was ever rejected, so the spread is that arm's rate.
        # An even-handed gate reads 0.0 here no matter how much it discards.
        self.assertAlmostEqual(adm["rate_spread"], 1 / rounds)

    def test_a_gate_that_eats_a_whole_arm_says_so_instead_of_dying_on_a_median(self):
        """Without the guard this is `StatisticsError: no median for empty data`.

        That message names neither the arm nor the gate, and it appears at the
        end of a long sweep. The totally one-sided gate is the case the
        admission columns exist for; it should not be the one that reports
        worst.
        """
        m = self._harness()

        def all_starved(binary, env_extra, timeout=1800):
            return {1: {"steady_ms": 1.0, "cold_ms": 2.0, "fnv": "abcd"}}, 0.10

        with unittest.mock.patch.object(m, "launch", all_starved):
            with self.assertRaises(SystemExit) as caught:
                m.prefill_matrix(2, 16, "shape", [1])
        message = str(caught.exception)
        self.assertIn("discarded every launch", message)
        for arm in ("before", "after", "aa"):
            self.assertIn(arm, message)

    def test_the_decode_matrix_names_a_starved_arm_too(self):
        """The guard has to exist on both matrices, not just the prefill one.

        Review caught the first version of this change claiming, in the
        report, that the *instrument* now names a starved arm, when the guard
        was only in `prefill_matrix`. A fully starved `before` or `aa` reached
        `ratio_stats` here and died on `no median for empty data` -- the exact
        error the claim said was retired -- and a fully starved `after` hit the
        generic "no parseable decode samples" return, which reads as a parsing
        bug rather than an admission one.
        """
        m = self._harness()

        def only_before_starved(binary, env_extra, timeout=1800):
            arm = Path(binary).name.replace("decode_", "")
            eff = 0.10 if arm == "before" else 0.99
            return {"steady": 1.0, "cold": 2.0, "checksum": "x", "raw": "r"}, eff

        with unittest.mock.patch.object(m, "decode_launch", only_before_starved):
            with self.assertRaises(SystemExit) as caught:
                m.decode_matrix(2, 16, 8)
        message = str(caught.exception)
        self.assertIn("discarded every decode launch", message)
        self.assertIn("before", message)


if __name__ == "__main__":
    unittest.main(verbosity=1)
