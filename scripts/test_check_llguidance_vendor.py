"""Source/archive provenance must bind the maintained and original identities."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import check_llguidance_vendor as audit


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        (base / "Cargo.toml").write_text('[workspace.package]\nversion="0.1.0-dev.6"\n')
        self.root = base / "vendor" / "llguidance"
        shutil.copytree(audit.DEFAULT_ROOT, self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_and_preserved_original_manifest_are_distinct(self):
        result = audit.verify(self.root)
        self.assertEqual(result["package"], "onnx-genai-llguidance")
        self.assertEqual(result["upstream"], "llguidance 1.9.1")
        (self.root / "UPSTREAM-Cargo.toml").write_text("tampered original")
        with self.assertRaisesRegex(RuntimeError, "preserved upstream metadata"):
            audit.verify(self.root)

    def test_owned_manifest_and_source_patches_cannot_drift(self):
        for name, message in [("Cargo.toml", "maintained compiler file"), ("src/lib.rs", "compiler patch hash")]:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n// unaudited change\n")
            with self.assertRaisesRegex(RuntimeError, message):
                audit.verify(self.root)
            path.write_bytes(original)

    def test_generated_or_unknown_files_are_not_source_provenance(self):
        for name in ["Cargo.toml.orig", "unexpected.txt"]:
            with self.subTest(name=name):
                path = self.root / name
                path.write_text("unreviewed generated file")
                with self.assertRaises(RuntimeError):
                    audit.verify(self.root)
                path.unlink()

    def package(self):
        original = (self.root / "Cargo.toml").read_text()
        (self.root / "Cargo.toml.orig").write_text(original)
        (self.root / "Cargo.toml").write_text(original.replace('version.workspace = true', 'version = "0.1.0-dev.6"').replace('repository.workspace = true', 'repository = "https://github.com/justinchuby/onnx-genai"'))
        (self.root / "Cargo.lock").write_text("version = 4\n")
        (self.root / ".cargo_vcs_info.json").write_text(json.dumps({"git":{"sha1":"a"*40},"path_in_vcs":"vendor/llguidance"}))

    def test_package_preserves_source_manifest_and_features(self):
        self.package()
        self.assertTrue(audit.verify(self.root, packaged=True)["packaged"])
        path = self.root / "Cargo.toml"
        path.write_text(path.read_text().replace('name = "onnx-genai-llguidance"', 'name = "llguidance"'))
        with self.assertRaisesRegex(RuntimeError, "identity/version/edition"):
            audit.verify(self.root, packaged=True)

    def test_package_cannot_change_audited_features_or_original_metadata(self):
        self.package()
        path = self.root / "Cargo.toml"
        path.write_text(path.read_text().replace('lark = []', 'lark = ["referencing"]'))
        with self.assertRaisesRegex(RuntimeError, "feature semantics"):
            audit.verify(self.root, packaged=True)
        path = self.root / "Cargo.toml.orig"
        path.write_text(path.read_text() + "\n# changed source manifest\n")
        with self.assertRaisesRegex(RuntimeError, "maintained compiler file"):
            audit.verify(self.root, packaged=True)


if __name__ == "__main__":
    unittest.main()
