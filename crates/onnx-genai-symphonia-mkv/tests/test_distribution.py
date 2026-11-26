"""Lightweight source/archive packaging checks; no Cargo or decoding is executed."""

import importlib.util
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import tomllib

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parents[1]
specification = importlib.util.spec_from_file_location(
    "distribution", CRATE / "verify_distribution.py"
)
distribution = importlib.util.module_from_spec(specification)
specification.loader.exec_module(distribution)


class DistributionTests(unittest.TestCase):
    def archive(
        self, directory, *, wheel=True, changes=None, omit=None, source_root=ROOT
    ):
        source = distribution.expected_source(source_root)
        version = distribution.source_context(source_root)[1]["package"]["version"]
        prefix = (
            f"onnx_genai/_third_party/{distribution.PACKAGE}-{version}/"
            if wheel
            else f"onnx_genai-0.1.0/crates/{distribution.PACKAGE}/"
        )
        source.update(changes or {})
        if omit:
            source.pop(omit)
        files = {prefix + name: value for name, value in source.items()}
        if wheel:
            files[prefix + "SOURCE_VERSION"] = (
                f"{distribution.PACKAGE} {version}\n".encode()
            )
            files["onnx_genai-0.1.0.dist-info/METADATA"] = (
                b"Name: onnx-genai\nVersion: 0.1.0\n"
            )
            path = Path(directory) / "test.whl"
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
        else:
            path = Path(directory) / "test.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for name, value in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
        return path

    def test_maturin_dependency_out_dir_uses_actual_untagged_wire_field(self):
        # Maturin v1.14.1 GlobPattern::WithOutDir has no serde rename_all.
        # Its public guide's crate-name spelling is silently ignored by serde;
        # the resulting root-crate fallback already failed an actual wheel build.
        # https://github.com/PyO3/maturin/blob/v1.14.1/src/pyproject_toml.rs#L52-L81
        for package, module in [
            ("onnx-genai-python", "onnx_genai"),
            ("onnx-genai-cli", "onnx_genai_server"),
        ]:
            project = tomllib.loads(
                (ROOT / "crates" / package / "pyproject.toml").read_text()
            )
            includes = project["tool"]["maturin"]["include"]
            selected = [
                entry
                for entry in includes
                if isinstance(entry, dict) and entry.get("from") == "out-dir"
            ]
            self.assertEqual(len(selected), 1, package)
            entry = selected[0]
            self.assertEqual(set(entry), {"path", "from", "to", "crate_name"}, package)
            self.assertEqual(entry["crate_name"], distribution.PACKAGE, package)
            self.assertEqual(entry["to"], f"{module}/_third_party", package)

    def test_complete_wheel_and_sdist_preserve_exact_source_and_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            for wheel in [True, False]:
                receipt = distribution.validate(
                    self.archive(directory, wheel=wheel), ROOT
                )
                self.assertIn("src/opus.rs", receipt["covered_files_sha256"])
                self.assertIn("build.rs", receipt["covered_files_sha256"])
                self.assertIn("NOTICE.md", receipt["covered_files_sha256"])

    def test_missing_notice_or_changed_covered_rust_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for wheel in [True, False]:
                with self.assertRaisesRegex(ValueError, "NOTICE.md"):
                    distribution.validate(
                        self.archive(directory, wheel=wheel, omit="NOTICE.md"), ROOT
                    )
                with self.assertRaisesRegex(ValueError, "src/opus.rs"):
                    distribution.validate(
                        self.archive(
                            directory, wheel=wheel, changes={"src/opus.rs": b"stale"}
                        ),
                        ROOT,
                    )

    def test_normalized_registry_manifest_requires_original_in_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            original = (CRATE / "Cargo.toml").read_bytes()
            workspace = distribution.source_context(ROOT)[1]["package"]
            normalized = original.decode()
            for key in ["version", "edition", "repository"]:
                normalized = normalized.replace(
                    f"{key}.workspace = true", f'{key} = "{workspace[key]}"'
                )
            changes = {"Cargo.toml": normalized.encode()}
            with self.assertRaisesRegex(ValueError, "Cargo.toml.orig"):
                distribution.validate(self.archive(directory, changes=changes), ROOT)
            changes["Cargo.toml.orig"] = original
            distribution.validate(self.archive(directory, changes=changes), ROOT)
            changes["Cargo.toml"] = normalized.replace(
                'license = "MPL-2.0"', 'license = "MIT"'
            ).encode()
            with self.assertRaisesRegex(ValueError, "package.license"):
                distribution.validate(self.archive(directory, changes=changes), ROOT)

    def test_registry_source_context_needs_no_enclosing_workspace_or_root_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "registry-source"
            expected = distribution.expected_source(ROOT)
            workspace = distribution.source_context(ROOT)[1]["package"]
            original = expected["Cargo.toml"]
            normalized = original.decode()
            for key in ["version", "edition", "repository"]:
                normalized = normalized.replace(
                    f"{key}.workspace = true", f'{key} = "{workspace[key]}"'
                )
            expected["Cargo.toml"] = normalized.encode()
            expected["Cargo.toml.orig"] = original
            for name, value in expected.items():
                path = registry / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            self.assertFalse((registry / "THIRD_PARTY_NOTICES.md").exists())
            for wheel in [True, False]:
                artifact = self.archive(directory, wheel=wheel, source_root=registry)
                distribution.validate(artifact, registry)

    def test_archive_path_traversal_is_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.archive(directory)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("../escaped", b"no")
            with self.assertRaisesRegex(ValueError, "unsafe/ambiguous"):
                distribution.validate(path, ROOT)


if __name__ == "__main__":
    unittest.main()
