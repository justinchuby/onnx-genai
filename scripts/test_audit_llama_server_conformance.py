"""Regression tests for honest conformance inventories and non-vacuous receipts."""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import audit_llama_server_conformance as audit


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())

    def test_reviewed_inventory_preserves_routes_and_original_acceptance(self):
        validated = audit.validate(audit.ROOT, self.inventory)
        self.assertEqual(len(validated["routes"]), 55)
        self.assertEqual(len(validated["acceptance"]), 9)
        self.assertEqual(
            sum(row["scope"] == "external_harness" for row in validated["routes"]), 4
        )
        report = audit.render(validated)
        self.assertIn("not fresh passing results", report)
        self.assertIn("destructive native-cache rollback", report)

    def test_missing_route_cannot_hide_behind_registration_audit(self):
        self.inventory["routes"].pop()
        with self.assertRaisesRegex(ValueError, "missing="):
            audit.validate(audit.ROOT, self.inventory)

    def test_missing_broad_requirement_is_not_a_complete_matrix(self):
        self.inventory["acceptance"].pop()
        with self.assertRaisesRegex(ValueError, "original acceptance categories"):
            audit.validate(audit.ROOT, self.inventory)

    def test_execution_exclusion_cannot_be_relabelled_as_server_work(self):
        row = next(
            row
            for row in self.inventory["routes"]
            if row["scope"] == "external_harness"
        )
        row["scope"] = "server"
        with self.assertRaisesRegex(ValueError, "external-harness boundary"):
            audit.validate(audit.ROOT, self.inventory)

    def test_renamed_test_and_helper_function_are_not_valid_source_links(self):
        for selector, message in [
            ("tests::nonexistent_conformance_test", "missing/ambiguous"),
            ("tests::tiny_state", "not an attributed test"),
        ]:
            inventory = copy.deepcopy(self.inventory)
            inventory["evidence"]["discovery"]["test"] = selector
            with self.assertRaisesRegex(ValueError, message):
                audit.validate(audit.ROOT, inventory)

    def test_deleted_assertion_marker_and_unknown_case_are_rejected(self):
        self.inventory["evidence"]["browser"]["source_markers"].append(
            "a removed browser assertion"
        )
        with self.assertRaisesRegex(ValueError, "assertion marker"):
            audit.validate(audit.ROOT, self.inventory)
        self.inventory["evidence"]["browser"]["source_markers"].pop()
        self.inventory["routes"][0]["evidence"] = ["case_that_was_not_added"]
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            audit.validate(audit.ROOT, self.inventory)

    def test_source_paths_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "invalid evidence source"):
            audit.source_path(audit.ROOT, "../../outside.rs")


class CargoContextTests(unittest.TestCase):
    def test_missing_implicit_configs_are_bound_and_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "project" / "workspace"
            root.mkdir(parents=True)
            environment = {"HOME": str(base / "home")}
            initial = audit.cargo_context(root, None, environment)
            local = root / ".cargo" / "config.toml"
            self.assertTrue(
                any(
                    item["path"] == str(local) and item["sha256"] is None
                    for group in initial["groups_low_to_high"]
                    for item in group["candidates"]
                )
            )
            local.parent.mkdir()
            local.write_text("[build]\njobs=1\n")
            appeared = audit.cargo_context(root, None, environment)
            self.assertNotEqual(initial, appeared)
            local.write_text("[build]\njobs=2\n")
            self.assertNotEqual(appeared, audit.cargo_context(root, None, environment))

    def test_home_override_legacy_precedence_and_relative_path_context(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            environment = {"HOME": str(base / "home"), "CARGO_HOME": "custom-cargo"}
            home = root / "custom-cargo"
            home.mkdir()
            (home / "config").write_text("[build]\njobs=1\n")
            (home / "config.toml").write_text("[build]\njobs=2\n")
            context = audit.cargo_context(root, None, environment)
            self.assertEqual(context["cargo_home"], str(home))
            self.assertEqual(
                context["groups_low_to_high"][0]["selected"], str(home / "config")
            )
            self.assertEqual(
                context["groups_low_to_high"][-1]["path_context"], str(root)
            )
            del environment["CARGO_HOME"]
            default = audit.cargo_context(root, None, environment)
            self.assertEqual(default["cargo_home"], str(base / "home" / ".cargo"))
            self.assertNotEqual(context, default)

    def test_explicit_symlink_binds_resolved_config_and_include_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            actual = root / "other" / "actual.toml"
            actual.parent.mkdir()
            actual.write_text('include = ["flags.toml"]\n')
            included = actual.parent / "flags.toml"
            included.write_text("[build]\njobs=1\n")
            # This decoy would be selected by the old symlink-directory bug.
            (root / "flags.toml").write_text("[build]\njobs=9\n")
            supplied = root / "link.toml"
            supplied.symlink_to(actual)
            environment = {"HOME": str(root / "home")}
            initial = audit.cargo_context(root, supplied, environment)
            explicit = initial["groups_low_to_high"][-1]
            self.assertEqual(explicit["supplied_path"], str(supplied))
            self.assertEqual(explicit["selected"], str(actual))
            self.assertEqual(explicit["path_context"], str(actual.parent.parent))
            self.assertEqual(initial["included_configs"][0]["path"], str(included))
            self.assertEqual(
                initial["included_configs"][0]["sha256"], audit.digest(included)
            )
            included.write_text("[build]\njobs=2\n")
            self.assertNotEqual(
                initial, audit.cargo_context(root, supplied, environment)
            )

    def test_optional_includes_and_build_environment_are_hashed_not_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            config = base / "config.toml"
            config.write_text(
                'include = [{path="optional.toml",optional=true}]\n[build]\njobs=1\n'
            )
            environment = {
                "HOME": str(base / "home"),
                "RUSTFLAGS": "-C opt-level=1",
                "CARGO_REGISTRY_TOKEN": "secret-not-for-receipt",
            }
            initial = audit.cargo_context(base, config, environment)
            self.assertFalse(initial["included_configs"][0]["exists"])
            serialized = json.dumps(initial)
            self.assertNotIn(environment["CARGO_REGISTRY_TOKEN"], serialized)
            self.assertNotIn(environment["RUSTFLAGS"], serialized)
            (base / "optional.toml").write_text("[build]\njobs=3\n")
            self.assertNotEqual(initial, audit.cargo_context(base, config, environment))
            environment["RUSTFLAGS"] = "-C opt-level=2"
            self.assertNotEqual(
                initial["environment_sha256"],
                audit.cargo_context(base, config, environment)["environment_sha256"],
            )


@unittest.skipUnless(os.name == "posix", "process group containment requires POSIX")
class ContainmentTests(unittest.TestCase):
    def worker(self, directory, interrupt=False, ignore_parent_term=False):
        # The worker waits for its child on TERM. Killing only the direct worker
        # cannot pass: its child would remain alive and fail the PID assertion.
        return "\n".join(
            [
                "import os, signal, subprocess, sys, time",
                "from pathlib import Path",
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])",
                f"Path({str(Path(directory) / 'child.pid')!r}).write_text(str(child.pid))",
                "def terminate(number, frame):",
                "    child.wait()",
                "    signal.signal(signal.SIGTERM,signal.SIG_IGN)"
                if ignore_parent_term
                else "    raise SystemExit(0)",
                "signal.signal(signal.SIGTERM,terminate)",
                "os.kill(os.getppid(),signal.SIGINT)" if interrupt else "pass",
                "time.sleep(60)",
            ]
        )

    def check_worker(self, interrupt=False, ignore_parent_term=False):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "child.pid"
            try:
                with (Path(directory) / "log").open("w") as output:
                    exception = (
                        audit.RunInterrupted if interrupt else subprocess.TimeoutExpired
                    )
                    with self.assertRaises(exception):
                        audit.run_contained(
                            [
                                sys.executable,
                                "-c",
                                self.worker(directory, interrupt, ignore_parent_term),
                            ],
                            cwd=Path(directory),
                            stdout=output,
                            timeout=2,
                            env=dict(os.environ),
                        )
                self.assertTrue(
                    pidfile.exists(),
                    "worker did not launch the descendant exercised by this regression",
                )
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), 0)
            finally:
                # A failing implementation must not leak the deliberately long
                # lived fixture process either.
                if pidfile.exists():
                    try:
                        os.kill(int(pidfile.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_signal_during_creation_is_deferred_until_child_handle_is_owned(self):
        process = MagicMock(pid=12345)
        process.wait.return_value = 0

        def create(*_args, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
            return process

        with (
            tempfile.TemporaryDirectory() as directory,
            (Path(directory) / "log").open("w") as output,
            patch.object(audit.subprocess, "Popen", side_effect=create),
            patch.object(audit.os, "killpg") as killpg,
            self.assertRaises(audit.RunInterrupted),
        ):
            try:
                audit.run_contained(
                    ["test-command"],
                    cwd=Path(directory),
                    stdout=output,
                    timeout=2,
                    env={},
                )
            finally:
                self.assertEqual(
                    killpg.call_args_list,
                    [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)],
                )
                self.assertTrue(process.wait.called)

    def test_timeout_terminates_descendant_and_reaps_worker(self):
        self.check_worker(ignore_parent_term=True)

    def test_interrupt_terminates_descendant_and_reaps_worker(self):
        self.check_worker(interrupt=True)


class ExecutionTests(unittest.TestCase):
    name = "test_scope::native_wire"

    def test_discovery_requires_exactly_one_test_identity(self):
        audit.listed_test(self.name + ": test\n\n1 test, 0 benchmarks\n", self.name)
        for output in [
            "",
            "0 tests, 0 benchmarks\n",
            self.name + ": test\n" + self.name + "_other: test\n",
            self.name + "_renamed: test\n",
        ]:
            with self.assertRaisesRegex(ValueError, "discovery selected"):
                audit.listed_test(output, self.name)

    def test_only_one_positive_unignored_result_can_pass(self):
        positive = f"test {self.name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n"
        audit.passed_test(positive, self.name)
        for output in [
            positive.replace("1 passed", "0 passed"),
            positive.replace("0 ignored", "1 ignored"),
            positive.replace("0 failed", "1 failed"),
            positive.replace("1 passed", "2 passed"),
            positive.replace(self.name, self.name + "_other"),
            positive + positive,
        ]:
            with self.assertRaisesRegex(ValueError, "exactly one passed"):
                audit.passed_test(output, self.name)

    def args(self, directory):
        return argparse.Namespace(
            receipt=Path(directory) / "receipt.json",
            run=["aliases_counts"],
            jobs=1,
            offline=True,
            no_default_features=False,
            features="",
            cargo_config=None,
            timeout_seconds=10,
        )

    def fake_cargo(self, command, **kwargs):
        if "--version" in command:
            kwargs["stdout"].write("tool-version\n")
            return subprocess.CompletedProcess(command, 0)
        name = command[command.index("--") - 1]
        if "--list" in command:
            output = name + ": test\n\n1 test, 0 benchmarks\n"
        else:
            output = f"test {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n"
        kwargs["stdout"].write(output)
        return subprocess.CompletedProcess(command, 0)

    def test_receipt_records_configuration_hashes_exact_commands_and_counts(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with (
                patch.object(
                    audit,
                    "source_state",
                    return_value={"revision": "a" * 40, "tree": "b" * 40},
                ),
                patch.object(
                    audit.subprocess, "check_output", return_value="tool-version\n"
                ),
                patch.object(audit, "run_contained", side_effect=self.fake_cargo),
            ):
                audit.run(inventory, args)
            receipt = json.loads(args.receipt.read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertIn(str(audit.INVENTORY), receipt["content_sha256"])
            self.assertEqual(receipt["features"], {"defaults": True, "additional": ""})
            self.assertEqual(receipt["cases"][0]["passed"], 1)
            self.assertIn("--list", receipt["cases"][0]["list"]["command"])
            self.assertIn("--exact", receipt["cases"][0]["run"]["command"])
            self.assertEqual(receipt["source"]["revision"], "a" * 40)

    def test_failed_or_changed_source_cannot_leave_a_success_receipt(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        for changed in [False, True]:
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                args = self.args(directory)
                original = {"revision": "a" * 40, "tree": "b" * 40}
                states = (
                    [original, {"revision": "c" * 40, "tree": "d" * 40}]
                    if changed
                    else [original]
                )
                state_patch = (
                    patch.object(audit, "source_state", side_effect=states)
                    if changed
                    else patch.object(audit, "source_state", return_value=original)
                )

                def cargo(command, current_changed=changed, **kwargs):
                    if current_changed:
                        return self.fake_cargo(command, **kwargs)
                    kwargs["stdout"].write("0 tests, 0 benchmarks\n")
                    return subprocess.CompletedProcess(command, 0)

                with (
                    state_patch,
                    patch.object(
                        audit.subprocess, "check_output", return_value="version\n"
                    ),
                    patch.object(audit, "run_contained", side_effect=cargo),
                    self.assertRaises(ValueError),
                ):
                    audit.run(inventory, args)
                receipt = json.loads(args.receipt.read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["cases"][0]["status"], "failed")
                self.assertNotIn("passed", receipt["cases"][0])

    def test_changed_external_config_invalidates_receipt(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.cargo_config = Path(directory) / "cargo-config.toml"
            args.cargo_config.write_text("[build]\njobs=1\n")

            def cargo(command, **kwargs):
                args.cargo_config.write_text("[build]\njobs=2\n")
                return self.fake_cargo(command, **kwargs)

            with (
                patch.object(
                    audit,
                    "source_state",
                    return_value={"revision": "a" * 40, "tree": "b" * 40},
                ),
                patch.object(
                    audit.subprocess, "check_output", return_value="version\n"
                ),
                patch.object(audit, "run_contained", side_effect=cargo),
                self.assertRaisesRegex(ValueError, "source/config changed"),
            ):
                audit.run(inventory, args)
            self.assertEqual(json.loads(args.receipt.read_text())["status"], "failed")

    def test_symlink_config_include_modification_invalidates_receipt(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            actual = Path(directory).resolve() / "other" / "actual.toml"
            actual.parent.mkdir()
            actual.write_text('include = ["flags.toml"]\n')
            included = actual.parent / "flags.toml"
            included.write_text("[build]\njobs=1\n")
            args.cargo_config = Path(directory).resolve() / "link.toml"
            args.cargo_config.symlink_to(actual)

            def cargo(command, **kwargs):
                if "--config" in command:
                    self.assertEqual(
                        command[command.index("--config") + 1], str(actual)
                    )
                included.write_text("[build]\njobs=2\n")
                return self.fake_cargo(command, **kwargs)

            with (
                patch.object(
                    audit,
                    "source_state",
                    return_value={"revision": "a" * 40, "tree": "b" * 40},
                ),
                patch.object(audit, "run_contained", side_effect=cargo),
                self.assertRaisesRegex(ValueError, "source/config changed"),
            ):
                audit.run(inventory, args)
            receipt = json.loads(args.receipt.read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["cases"][0]["status"], "failed")

    def test_changed_implicit_config_invalidates_receipt(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            home = Path(directory) / "home"

            def cargo(command, **kwargs):
                config = home / ".cargo" / "config.toml"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_text("[build]\njobs=2\n")
                return self.fake_cargo(command, **kwargs)

            with (
                patch.dict(os.environ, {"HOME": str(home)}, clear=True),
                patch.object(
                    audit,
                    "source_state",
                    return_value={"revision": "a" * 40, "tree": "b" * 40},
                ),
                patch.object(
                    audit.subprocess, "check_output", return_value="version\n"
                ),
                patch.object(audit, "run_contained", side_effect=cargo),
                self.assertRaisesRegex(ValueError, "source/config changed"),
            ):
                audit.run(inventory, args)
            receipt = json.loads(args.receipt.read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["cases"][0]["status"], "failed")

    def test_keyboard_interrupt_marks_receipt_failed(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with (
                patch.object(
                    audit,
                    "source_state",
                    return_value={"revision": "a" * 40, "tree": "b" * 40},
                ),
                patch.object(
                    audit.subprocess, "check_output", return_value="version\n"
                ),
                patch.object(
                    audit, "run_contained", side_effect=KeyboardInterrupt("cancelled")
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                audit.run(inventory, args)
            receipt = json.loads(args.receipt.read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("KeyboardInterrupt", receipt["error"])

    def test_browser_or_empty_run_is_not_silently_passed(self):
        inventory = json.loads((audit.ROOT / audit.INVENTORY).read_text())
        with tempfile.TemporaryDirectory() as directory:
            for ids in [[], ["browser"]]:
                args = self.args(directory)
                args.run = ids
                with self.assertRaises(ValueError):
                    audit.run(inventory, args)
                self.assertFalse(args.receipt.exists())


if __name__ == "__main__":
    unittest.main()
