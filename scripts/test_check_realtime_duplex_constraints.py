#!/usr/bin/env python3
"""Lightweight evidence-binding regressions; never invokes Cargo."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_realtime_duplex_constraints as driver


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.tree = self.base / "source"
        self.tree.mkdir()
        for name in [
            "Cargo.toml", "Cargo.lock",
            "crates/onnx-genai-server/Cargo.toml",
            "crates/onnx-genai-engine/Cargo.toml",
            "crates/onnx-genai-server/src/realtime.rs",
            "crates/onnx-genai-server/src/realtime/tools.rs",
            "crates/onnx-genai-server/src/realtime/websocket.rs",
            "crates/onnx-genai-engine/src/realtime.rs",
            "crates/onnx-genai-engine/src/byte_grammar.rs",
            "crates/onnx-genai-engine/src/other.rs",
        ]:
            path = self.tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("initial\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit()
        self.environment = dict(os.environ, HOME=str(self.base / "home"), CARGO_HOME=str(self.base / "cargo"))

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.tree), *arguments], text=True).strip()

    def commit(self):
        self.git("-c", "user.name=Binding Test", "-c", "user.email=binding@example.invalid", "commit", "--quiet", "-am", "snapshot")

    def snapshot(self, **kwargs):
        return driver.source_identity(self.tree, None, self.environment, **kwargs)

    def test_clean_tree_binds_all_tracked_source_and_rejects_dirty_or_new_files(self):
        initial = self.snapshot()
        self.assertIn("Cargo.lock", initial["source_sha256"])
        self.assertIn("crates/onnx-genai-engine/Cargo.toml", initial["source_sha256"])
        # An engine file outside the explicit digest list is still bound by the
        # clean tree hash. Dirty changes reject; committing changes differs.
        other = self.tree / "crates/onnx-genai-engine/src/other.rs"
        other.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "clean worktree"):
            self.snapshot()
        self.commit()
        with self.assertRaisesRegex(ValueError, "changed during validation"):
            driver.require_unchanged(initial, self.snapshot())
        (self.tree / "untracked.rs").write_text("new\n")
        with self.assertRaisesRegex(ValueError, "clean worktree"):
            self.snapshot()

    def test_baseline_requires_detached_clean_exact_revision(self):
        revision = self.git("rev-parse", "HEAD")
        with patch.object(driver, "BASELINE_REVISION", revision):
            with self.assertRaisesRegex(ValueError, "detached"):
                self.snapshot(baseline=True)
            self.git("checkout", "--quiet", "--detach")
            self.assertIsNone(self.snapshot(baseline=True)["symbolic_head"])
            self.git("checkout", "--quiet", "-b", "same-revision-branch")
            with self.assertRaisesRegex(ValueError, "detached"):
                self.snapshot(baseline=True)
            self.git("checkout", "--quiet", "--detach")
            (self.tree / "Cargo.lock").write_text("dirty\n")
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                self.snapshot(baseline=True)

    def test_implicit_config_appearance_modification_and_environment_invalidate(self):
        initial = self.snapshot()
        config = Path(self.environment["CARGO_HOME"]) / "config.toml"
        config.parent.mkdir()
        config.write_text("[build]\njobs=1\n")
        appeared = self.snapshot()
        with self.assertRaisesRegex(ValueError, "changed during validation"):
            driver.require_unchanged(initial, appeared)
        config.write_text("[build]\njobs=2\n")
        with self.assertRaisesRegex(ValueError, "changed during validation"):
            driver.require_unchanged(appeared, self.snapshot())
        current = self.snapshot()
        self.environment["RUSTFLAGS"] = "--cfg binding_changed"
        updated = self.snapshot()
        with self.assertRaisesRegex(ValueError, "changed during validation"):
            driver.require_unchanged(current, updated)
        self.assertNotIn("--cfg binding_changed", str(updated))

    def test_mutation_during_command_refuses_verified_evidence(self):
        import sys
        source = self.tree / "crates/onnx-genai-engine/src/other.rs"
        for change in ["source", "config", "environment"]:
            with self.subTest(change=change):
                def process(command, change=change, **kwargs):
                    if change == "source":
                        source.write_text("changed during command\n")
                    elif change == "config":
                        config = Path(self.environment["CARGO_HOME"]) / "config.toml"
                        config.parent.mkdir(exist_ok=True)
                        config.write_text("[build]\njobs=7\n")
                    else:
                        os.environ["RUSTFLAGS"] = "--cfg changed_during_command"
                    kwargs["stdout"].write("not accepted as evidence\n")
                    return subprocess.CompletedProcess(command, 0)

                arguments = [
                    "driver", "--fixed-only", "--fixed-tree", str(self.tree),
                    "--target-dir", str(self.base / "target"),
                    "--log-dir", str(self.base / "logs"),
                ]
                with (
                    patch.dict(os.environ, self.environment, clear=True),
                    patch.object(sys, "argv", arguments),
                    patch.object(driver, "run_contained", side_effect=process) as run,
                    self.assertRaises(ValueError),
                ):
                    driver.main()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(list((self.base / "logs").glob("*/verified.json")), [])
                source.write_text("initial\n")


if __name__ == "__main__":
    unittest.main()
