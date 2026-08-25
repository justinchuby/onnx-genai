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

**What it does not catch**, written down so nobody reads more into a green
run than is there:

- A driver **outside this directory**. `crates/onnx-runtime-ep-cpu/benches/
  acc0_*.py` are ungated and are the rest of #2043.
- A driver **misdeclared as a generator** whose binary path is built at
  runtime (`os.path.join("target", "release", ...)`, an f-string, an env
  var). The contradiction check reads literals, so it would not object --
  `ab.py` itself has no `target/` literal. Declaring a driver a generator is
  a false statement in a reviewed file, which is the layer that catches it.

Run: `python3 scripts/ort_ab/test_gate_conformance.py`
"""

from __future__ import annotations

import ast
import re
import unittest
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


if __name__ == "__main__":
    unittest.main(verbosity=1)
