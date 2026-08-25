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
else must not look like a driver -- if it spawns a benchmark or opens an
inference session, the declaration is contradicted by the file itself and the
suite fails.

The point is not that this catches today's drivers; #2043 already listed
those. It is that the *next* harness cannot ship ungated by accident, because
an unclassified file is a failure and a misclassified one is a failure with a
reason attached.

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


# Both import styles reach the same gate: `hostlock_gate.require(...)` and
# the `from hostlock_gate import require` that `ab.py` uses. Matching only the
# qualified form would have reported the one driver that has taken the lock
# since #2032 as ungated.
GATE_CALLS = (
    re.compile(r"hostlock_gate\.require\("),
    re.compile(r"from hostlock_gate import[^\n]*(?:\n[^)]*)?\brequire\b"),
)


def takes_the_lock(source: str) -> bool:
    if not any(mark.search(source) for mark in GATE_CALLS):
        return False
    # The import alone is not the gate: a driver that imports `require` and
    # never calls it is exactly as unprotected as one that does not import it.
    return re.search(r"^\s*(?:lock_label|_)?[^\n]*\brequire\(", source, re.M) is not None


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
        if looks_like_a_driver(source, binary_paths=name not in TESTS):
            out.append(
                f"{name}: declared not-a-driver, but it starts a benchmark or "
                "opens an inference session"
            )
    return out


def read_sources() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(ORT_AB.glob("*.py"))}


class Classification(unittest.TestCase):
    def test_every_file_in_the_directory_has_exactly_one_role(self):
        # A new harness lands here as an unclassified file, which fails --
        # rather than as a silently ungated one, which does not.
        self.assertEqual(unclassified(sorted(read_sources())), [])

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
        text = self.WORKFLOW.read_text()
        self.assertIn('"scripts/ort_ab/**"', text)
        # Both triggers: a pull_request-only filter would let a direct push
        # to main land an ungated driver unchecked.
        self.assertEqual(text.count('"scripts/ort_ab/**"'), 2)

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

    def test_a_declared_driver_is_not_reported_as_contradicted(self):
        # Otherwise every driver would be reported twice, and the two findings
        # mean different things.
        self.assertEqual(contradicted({"ab.py": "import onnxruntime\n"}), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
