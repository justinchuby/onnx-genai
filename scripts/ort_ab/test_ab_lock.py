#!/usr/bin/env python3
"""Tests for `ab.py`'s host-lock admission.

The interesting half is not that a locked run is allowed -- it is that each
*unprotected* shape is refused, and refused distinguishably. A gate that
allowed one of them would put load on a host somebody else had declared, and
the resulting numbers would carry a `host_lock` label saying otherwise.

The end-to-end cells drive the real `hostlock.sh` and a stub arm binary that
prints a result line and exits. They cost no measurable CPU: nothing is
benchmarked, which is what lets this run in CI on a shared runner.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab import (  # noqa: E402
    ancestry,
    lock_verdict,
    lock_columns,
    parse_provenance,
    read_provenance,
    window_label,
)

AB = Path(__file__).resolve().parent / "ab.py"
SWEEP = Path(__file__).resolve().parent / "sweep_decode.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_decode  # noqa: E402
HOSTLOCK = Path(__file__).resolve().parents[1] / "hostlock.sh"

# A stub arm: prints the line `ab.py` parses and exits. Using a real benchmark
# here would make the admission test cost minutes and a quiet host, which is
# the thing the lock exists to ration.
STUB_ARM = """#!/bin/sh
echo "native=1.000 ms ort=2.000 ms native/ort=0.500 native_p90=1.1 ort_p90=2.1 \
native_min=0.9 ort_min=1.9 native_spread=0.2 ort_spread=0.2 parity=PASS"
"""


def prov(state: str, pid: str = "none", owner: str = "none") -> dict[str, str]:
    return {
        "hostlock_state": state,
        "held_by": owner,
        "held_pid": pid,
        "runnable": "3",
        "contended": "no",
    }


class Verdict(unittest.TestCase):
    """The decision table, as a table."""

    def test_a_declaration_held_by_an_ancestor_admits_the_run(self):
        label, refusal = lock_verdict(prov("HELD", "4242", "leon"), {1, 99, 4242})
        self.assertIsNone(refusal)
        self.assertEqual(label, "mine:leon")

    def test_a_declaration_held_by_a_peer_stops_the_run(self):
        label, refusal = lock_verdict(prov("HELD", "4242", "roy"), {1, 99})
        self.assertIsNotNone(refusal)
        self.assertEqual(label, "foreign:roy")

    def test_a_lock_held_by_a_child_does_not_cover_the_matrix(self):
        """The incident this gate exists for.

        One `hostlock.sh run` per benchmark child releases the lock between
        arms, so the host reads idle in the gap and a peer starts a sweep in
        the middle of somebody's interleaved A/B. A child's pid is not in our
        ancestry, so the shape is refused even though a lock is genuinely
        held.
        """
        mine = os.getpid()
        child = mine + 1000000  # not an ancestor of anything we are
        _, refusal = lock_verdict(prov("HELD", str(child), "leon"), ancestry(mine))
        self.assertIsNotNone(refusal)

    def test_every_unprotected_state_is_refused_and_labelled_distinctly(self):
        # The mapping is the specification. Two states share the `unknown`
        # label on purpose -- an empty reading and a literal UNKNOWN are the
        # same fact -- and every other one is distinct, because a reader has to
        # be able to tell a host that had no lock from one whose lock died
        # under it.
        expected = {
            "FREE": "free",
            "STALE": "stale:roy",
            "EXPIRED": "expired:roy",
            "UNUSABLE": "unusable",
            "UNKNOWN": "unknown",
            "": "unknown",
        }
        for state, want in expected.items():
            label, refusal = lock_verdict(prov(state, "7", "roy"), {7})
            self.assertIsNotNone(refusal, f"{state} must not admit a run")
            self.assertEqual(label, want, state)
        self.assertEqual(
            len(set(expected.values())),
            5,
            "collapsing any further would hide a distinction the run needs",
        )

    def test_an_unreadable_lock_is_refused_rather_than_assumed_free(self):
        label, refusal = lock_verdict({}, {1})
        self.assertEqual(label, "unknown")
        self.assertIsNotNone(refusal)

    def test_a_broken_hostlock_reads_as_unreadable_not_as_free(self):
        def explode(*_a, **_kw):
            raise OSError("no such tool")

        self.assertEqual(read_provenance(runner=explode), {})


class Window(unittest.TestCase):
    """The label has to describe the whole run, not one end of it."""

    def test_a_steady_holder_keeps_the_label(self):
        before = prov("HELD", "17", "leon")
        self.assertEqual(window_label("mine:leon", before, dict(before)), "mine:leon")

    def test_a_failed_second_reading_is_not_a_handoff(self):
        """`changed` asserts custody moved. An unreadable end says nothing.

        Stamping `changed` for a lock that could not be re-read blames a
        completed, genuinely protected matrix for a handoff that never
        happened -- and, worse, makes a real handoff indistinguishable from a
        60-second timeout on the final read.
        """
        before = prov("HELD", "17", "leon")
        self.assertEqual(window_label("mine:leon", before, {}), "unverified-end")

    def test_a_lock_that_changed_hands_is_not_reported_as_held_throughout(self):
        before = prov("HELD", "17", "leon")
        for after in (
            prov("FREE"),
            prov("HELD", "17", "roy"),
            prov("HELD", "18", "leon"),
        ):
            self.assertEqual(
                window_label("mine:leon", before, after),
                "changed",
                after,
            )


class Columns(unittest.TestCase):
    """The provenance-key to column-name mapping, as a table.

    Every value below is distinct on purpose: two columns reading each
    other's provenance key, or one reading a plausible neighbour, produces a
    row whose numbers sit under names that do not describe them. A cell that
    only checked the keys were present could not tell.
    """

    def test_each_column_carries_the_field_it_is_named_after(self):
        fields = {
            "held_by": "leon",
            "held_pid": "4242",
            "runnable": "7",
            "held_secs": "99",
            "contended": "no",
            "hostlock_state": "HELD",
        }
        self.assertEqual(
            lock_columns("mine:leon", fields),
            {
                "host_lock": "mine:leon",
                "lock_owner": "leon",
                "lock_anchor_pid": "4242",
                "runnable_at_start": "7",
            },
        )

    def test_a_missing_reading_stamps_placeholders_rather_than_blanks(self):
        # An empty cell in a CSV reads as "not applicable". These rows always
        # have an answer, even when the answer is that there was none.
        self.assertEqual(
            lock_columns("unknown", {}),
            {
                "host_lock": "unknown",
                "lock_owner": "none",
                "lock_anchor_pid": "none",
                "runnable_at_start": "unknown",
            },
        )


class Ancestry(unittest.TestCase):
    def test_the_walk_terminates_on_a_cycle(self):
        # A reparented or namespaced process can report a parent already in
        # the chain. Without the seen-set this loops until `limit`, and with a
        # larger limit it would hang the harness before it ran anything.
        self.assertEqual(ancestry(5, parent=lambda pid: 5), {5})

    def test_the_walk_stops_at_init(self):
        chain = {5: 4, 4: 1, 1: 0}
        self.assertEqual(ancestry(5, parent=lambda pid: chain.get(pid)), {1, 4, 5})

    def test_a_real_chain_contains_this_process_and_its_parent(self):
        chain = ancestry(os.getpid())
        self.assertIn(os.getpid(), chain)
        self.assertIn(os.getppid(), chain)


class Parsing(unittest.TestCase):
    def test_the_oneline_format_round_trips(self):
        text = "hostlock_state=HELD held_by=leon held_pid=17 runnable=4 reason=x"
        self.assertEqual(parse_provenance(text)["held_by"], "leon")
        self.assertEqual(parse_provenance(text)["held_pid"], "17")

    def test_the_real_script_emits_the_keys_this_gate_reads(self):
        """Against the script, not against my memory of it.

        A key rename in `hostlock.sh` would otherwise leave the gate reading
        `hostlock_state` forever, finding nothing, and -- because the gate is
        fail-closed -- refusing every run on a correctly locked host. The
        failure would be loud, but it would look like a lock bug rather than a
        parser one.
        """
        with tempfile.TemporaryDirectory(dir=str(AB.parent)) as tmp:
            env = dict(os.environ, HOSTLOCK_DIR=f"{tmp}/hl", HOSTLOCK_PRIVATE_OK="1")
            out = subprocess.run(
                ["bash", str(HOSTLOCK), "provenance", "--oneline"],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            fields = parse_provenance(out.stdout)
        for key in ("hostlock_state", "held_by", "held_pid", "runnable", "contended"):
            self.assertIn(key, fields, out.stdout)


class EndToEnd(unittest.TestCase):
    """`ab.py` invoked for real, with a stub arm and a scratch lock."""

    def setUp(self):
        # Under the repo, never the shared lock and never a temp dir outside
        # it: a test that used the real lock directory could release a lock a
        # colleague was relying on.
        self.tmp = tempfile.mkdtemp(dir=str(AB.parent))
        self.arm = Path(self.tmp) / "arm.sh"
        self.arm.write_text(STUB_ARM)
        self.arm.chmod(0o755)
        self.csv = Path(self.tmp) / "out.csv"
        self.env = dict(
            os.environ,
            HOSTLOCK_DIR=f"{self.tmp}/hl",
            HOSTLOCK_PRIVATE_OK="1",
        )

    def tearDown(self):
        subprocess.run(["rm", "-rf", self.tmp], check=False)

    def ab_args(self):
        return [
            sys.executable,
            str(AB),
            "--arms",
            f"a={self.arm}",
            "--models",
            "fixture.onnx",
            "--threads",
            "1",
            "--trials",
            "1",
            "--runs",
            "1",
            "--warmups",
            "0",
            "--csv",
            str(self.csv),
        ]

    def test_an_unlocked_matrix_is_refused_before_any_arm_runs(self):
        out = subprocess.run(
            self.ab_args(), capture_output=True, text=True, env=self.env, timeout=300
        )
        self.assertEqual(out.returncode, 3, out.stdout + out.stderr)
        self.assertIn("refusing to measure", out.stderr)
        self.assertIn("hostlock.sh run --owner", out.stderr)
        self.assertFalse(
            self.csv.exists(), "a refused run must not leave a result file"
        )

    def test_the_wrapped_invocation_is_admitted_and_stamps_every_row(self):
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon-selftest",
                "--reason",
                "ab-admission-selftest",
                "--",
                *self.ab_args(),
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=300,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        # By column name and value, not by substring: a rename to
        # `host_lock_unused` leaves the owner in the file and every
        # `"host_lock" in text` assertion green while nothing downstream can
        # find the field.
        rows = list(csv.DictReader(self.csv.read_text().splitlines()))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["host_lock"], "mine:leon-selftest", row)
            self.assertEqual(row["lock_owner"], "leon-selftest", row)
            self.assertNotEqual(row["lock_anchor_pid"], "none", row)
            self.assertTrue(row["runnable_at_start"].isdigit(), row)

    def test_an_arm_with_env_overrides_still_runs(self):
        """The gate refactor deleted `import os`, and only this path uses it.

        Every other cell passes an arm with an empty override dict, which is
        falsy, so the `dict(os.environ)` line is never reached and a missing
        import is invisible. `--arm-env` is the documented way to A/B a
        feature flag with one binary in both arms -- the most common use of
        this harness -- and it raised `NameError` after taking the lock and
        before writing a single row.
        """
        args = [*self.ab_args(), "--arm-env", "a=ONNX_GENAI_TEST_FLAG=1"]
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon",
                "--reason",
                "arm-env regression",
                "--",
                *args,
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertNotIn("NameError", out.stderr)
        with self.csv.open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        self.assertEqual(rows[0]["host_lock"], "mine:leon", rows[0])

    def test_unlocked_does_not_override_a_peers_declaration(self):
        """The escape hatch protects our labels, not their run.

        `--unlocked` exists so an unprotected run cannot be quoted later as a
        protected one. That reasoning is entirely about our own rows, and it
        does not extend to a box somebody else has declared: the damage there
        lands on their measurement, where no label of ours can reach it.
        """
        anchor = subprocess.Popen(["sleep", "120"])
        try:
            taken = subprocess.run(
                [
                    "bash",
                    str(HOSTLOCK),
                    "acquire",
                    "--owner",
                    "someone-else",
                    "--reason",
                    "peer-declaration",
                    "--pid",
                    str(anchor.pid),
                ],
                capture_output=True,
                text=True,
                env=self.env,
                timeout=300,
            )
            self.assertEqual(taken.returncode, 0, taken.stdout + taken.stderr)
            out = subprocess.run(
                [*self.ab_args(), "--unlocked"],
                capture_output=True,
                text=True,
                env=self.env,
                timeout=300,
            )
            self.assertEqual(out.returncode, 3, out.stdout + out.stderr)
            self.assertIn("someone-else", out.stderr)
            self.assertFalse(self.csv.exists())
        finally:
            anchor.kill()
            anchor.wait()

    def test_unlocked_by_request_runs_but_marks_the_rows(self):
        out = subprocess.run(
            [*self.ab_args(), "--unlocked"],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=300,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("WARNING: running unlocked", out.stderr)
        rows = list(csv.DictReader(self.csv.read_text().splitlines()))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(
                row["host_lock"],
                "unlocked:free",
                "the escape hatch must leave the rows self-identifying",
            )


# `${WIDTH_FIELDS-...}` so a cell can drop or rewrite the realized-width
# report without a second stub: an older binary that never emitted them is a
# case the sweep has to survive, and it is one `env` away from here.
STUB_BENCH = """#!/bin/sh
# Emits one line in `bench_generic`'s paired format. The numbers are fixed:
# nothing here measures anything, it only has to parse.
W="${WIDTH_FIELDS-native_pool_width=1 native_path=flat native_task_width=1 \\
native_cpus=32 native_width_as_requested=yes}"
echo "shape=decode native=1.000 ms ort=2.000 ms native/ort=0.500 \\
native_p90=1.100 ort_p90=2.200 native_min=0.900 ort_min=1.800 $W"
"""


class EndOfWindow(unittest.TestCase):
    """A streamed driver cannot relabel printed rows, so the exit code is it."""

    def test_a_handoff_is_a_failure_and_says_to_discard(self):
        code, msg = sweep_decode.end_of_window_verdict("mine:leon", "changed")
        self.assertEqual(code, 4)
        self.assertIn("discard", msg)

    def test_an_unreadable_end_is_reported_without_claiming_a_handoff(self):
        # The distinction `window_label` exists to preserve: a failed read is
        # not evidence of a takeover, and telling someone to throw away a
        # sound matrix on that basis is its own defect.
        code, msg = sweep_decode.end_of_window_verdict("mine:leon", "unverified-end")
        self.assertEqual(code, 5)
        self.assertNotIn("discard", msg)
        self.assertNotIn("spans the change", msg)
        self.assertIn("unverified", msg)

    def test_an_unchanged_window_is_silent(self):
        self.assertEqual(
            sweep_decode.end_of_window_verdict("mine:leon", "mine:leon"), (0, None)
        )


class WidthVerdict(unittest.TestCase):
    """Does the row describe the route its `t` column names?

    A thread sweep whose leftmost column takes a serial short-circuit is not
    a scaling curve, and the harness printed four such rows before anybody
    noticed -- because `--native-threads 1` was a request the table reported
    back as though it were a measurement.

    The check is categorical, so it holds on a busy shared host: the lanes
    either came back or they did not. It is not a quiet-host requirement and
    must not become one (#1802).
    """

    def cells(self, *answers, path="pool", pool="4"):
        return [
            {"as_requested": a, "path": path, "pool_width": pool} for a in answers
        ]

    def test_a_realized_request_is_silent(self):
        self.assertEqual(
            sweep_decode.width_verdict(4, self.cells("yes", "yes")), ("yes", None)
        )

    def test_a_reduced_width_is_named_not_comparable(self):
        label, why = sweep_decode.width_verdict(
            16, self.cells("no", "no", path="flat", pool="1")
        )
        self.assertEqual(label, "no")
        self.assertIn("not", why)
        self.assertIn("comparable", why)
        self.assertIn("flat", why)

    def test_a_binary_that_cannot_report_width_is_not_read_as_agreement(self):
        # `absent` and `no` are different findings: one says the request was
        # reduced, the other says nothing here can tell you either way.
        label, why = sweep_decode.width_verdict(8, self.cells("absent"))
        self.assertEqual(label, "absent")
        self.assertIn("unverified", why)

    def test_a_request_that_never_reached_the_engine_says_so(self):
        label, why = sweep_decode.width_verdict(8, self.cells("n/a"))
        self.assertEqual(label, "not-requested")
        self.assertIn("did not reach", why)

    def test_trials_that_disagree_are_not_medianed_silently(self):
        label, why = sweep_decode.width_verdict(8, self.cells("yes", "no"))
        self.assertEqual(label, "varied")
        self.assertIn("not all on the same route", why)

    def test_the_width_fields_are_read_off_a_real_result_line(self):
        line = (
            "result: native=1.0 ms ort=2.0 ms native/ort=0.5 native_p90=1.1 "
            "ort_p90=2.1 native_min=0.9 ort_min=1.9 native_threads=4 "
            "native_pool_width=2 native_path=pool native_task_width=2 "
            "native_cpus=32 native_width_as_requested=no arm=x parity=PASS"
        )
        self.assertEqual(
            sweep_decode.width_report(line),
            {
                "pool_width": "2",
                "path": "pool",
                "task_width": "2",
                "cpus": "32",
                "as_requested": "no",
            },
        )

    def test_a_line_without_the_fields_reports_absent_rather_than_guessing(self):
        self.assertEqual(
            sweep_decode.width_report("result: native=1.0 ms ort=2.0 ms")[
                "as_requested"
            ],
            "absent",
        )


class SweepDecodeGate(unittest.TestCase):
    """The thread sweep is the most saturating thing we run.

    It is therefore the one driver where "somebody forgot the wrapper" costs
    the most, both to the run and to whoever else is on the box.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir=str(SWEEP.parent))
        self.bench = Path(self.tmp) / "bench.sh"
        self.bench.write_text(STUB_BENCH)
        self.bench.chmod(0o755)
        self.env = dict(
            os.environ, HOSTLOCK_DIR=f"{self.tmp}/hl", HOSTLOCK_PRIVATE_OK="1"
        )

    def tearDown(self):
        subprocess.run(["rm", "-rf", self.tmp], check=False)

    def args(self):
        return [
            sys.executable,
            str(SWEEP),
            "--binary",
            str(self.bench),
            "--models",
            "m.onnx",
            "--threads",
            "1",
            "--trials",
            "1",
            "--runs",
            "1",
            "--warmups",
            "0",
        ]

    def test_a_sweep_whose_lock_changed_hands_reports_it_and_fails(self):
        """The rows are already printed, so the only honest move is to fail.

        A sweep whose custody moved is not half-good data: the thread counts
        above and below the change were compared across it. The fixture
        rewrites `owner=` in the scratch lock while the stub bench runs, which
        is what a real takeover looks like to the end-of-window read.
        """
        handoff = Path(self.tmp) / "handoff.sh"
        handoff.write_text(
            "#!/bin/sh\n"
            'sed -i "s/^owner=.*/owner=someone-else/" "$HOSTLOCK_DIR/meta"\n'
            + STUB_BENCH.split("\n", 1)[1]
        )
        handoff.chmod(0o755)
        args = self.args()
        args[args.index("--binary") + 1] = str(handoff)
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon",
                "--reason",
                "handoff fixture",
                "--",
                *args,
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=600,
        )
        self.assertIn("host_lock=changed", out.stderr)
        self.assertIn("discard", out.stderr)
        # 4 exactly. A caller running several matrices has to tell "the lock
        # changed hands" from "the binary crashed" (1) and from "the host was
        # not mine" (3), and `!= 0` pins none of that.
        self.assertEqual(out.returncode, 4, out.stdout + out.stderr)

    def test_a_cell_whose_width_was_not_realized_fails_the_sweep(self):
        """The table still prints -- and the exit code stops it being quoted.

        Roy published four decode rows at width 1 before learning that width 1
        takes a serial short-circuit rather than a one-worker pool. Both arms
        of his A/B were on that path, so the comparison stood, but the *scope*
        did not: it was a serial-path result labelled as a decode result. The
        row is worth seeing; leaving with exit 0 is what makes it quotable.
        """
        env = dict(
            self.env,
            WIDTH_FIELDS=(
                "native_pool_width=1 native_path=flat native_task_width=1 "
                "native_cpus=32 native_width_as_requested=no"
            ),
        )
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon",
                "--reason",
                "width non-vacuity",
                "--",
                *self.args(),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertEqual(out.returncode, 6, out.stdout + out.stderr)
        self.assertIn("not comparable", out.stderr)
        self.assertIn("not a scaling curve", out.stderr)
        row = next(
            l for l in out.stdout.splitlines() if l.split() and l.split()[0] == "m"
        )
        # Second from the right: the width answer sits beside the lock label,
        # because a row needs both to be worth anything -- who held the box,
        # and whether the column describes the route.
        self.assertEqual(row.split()[-2], "no")

    def test_a_realized_width_reaches_the_row_and_the_header(self):
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon",
                "--reason",
                "width realized",
                "--",
                *self.args(),
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        header = next(l for l in out.stdout.splitlines() if "ratio_min" in l).split()
        self.assertEqual(header[-2], "width_ok")
        row = next(
            l for l in out.stdout.splitlines() if l.split() and l.split()[0] == "m"
        )
        self.assertEqual(row.split()[-2], "yes")

    def test_the_sweeps_escape_hatch_runs_and_stamps_every_row(self):
        """Otherwise hard-wiring `unlocked=False` would be invisible.

        The flag is what makes an unprotected smoke test *say* it was
        unprotected, so a suite that never exercises it cannot tell the
        difference between "refuses correctly" and "refuses always".
        """
        out = subprocess.run(
            [*self.args(), "--unlocked"],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=300,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("WARNING: running unlocked", out.stderr)
        row = next(
            l for l in out.stdout.splitlines() if l.split() and l.split()[0] == "m"
        )
        self.assertEqual(row.split()[-1], "unlocked:free")

    def test_an_unlocked_sweep_refuses_before_it_starts_any_child(self):
        out = subprocess.run(
            self.args(), capture_output=True, text=True, env=self.env, timeout=300
        )
        self.assertEqual(out.returncode, 3, out.stdout + out.stderr)
        self.assertIn("refusing to measure", out.stderr)
        self.assertIn("sweep_decode.py", out.stderr)
        # The remedy has to name the driver, not ab.py -- a wrapper someone
        # pastes for the wrong script is a wrapper that does not span the arms.
        self.assertNotIn("ab.py", out.stderr)
        self.assertEqual(out.stdout, "")

    def test_a_locked_sweep_runs_and_every_row_carries_the_label(self):
        out = subprocess.run(
            [
                "bash",
                str(HOSTLOCK),
                "run",
                "--owner",
                "leon",
                "--reason",
                "sweep gate test",
                "--",
                *self.args(),
            ],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        # `hostlock.sh run` prints its own outcome first, so the driver's
        # output is found by content rather than by position.
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        banner = next(l for l in lines if l.startswith("host_lock="))
        self.assertIn("host_lock=mine:leon", banner)
        self.assertIn("lock_owner=leon", banner)
        header = next(l for l in lines if "ratio_min" in l).split()
        self.assertEqual(header[-1], "host_lock")
        # The label is on the DATA row, not only in a banner: a banner is lost
        # the moment somebody pastes one interesting row into a doc.
        row = next(l for l in lines if l.split()[0] == "m")
        self.assertEqual(row.split()[-1], "mine:leon")
        self.assertEqual(len(row.split()), len(header))


if __name__ == "__main__":
    unittest.main()
