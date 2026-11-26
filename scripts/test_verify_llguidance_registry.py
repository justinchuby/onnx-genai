"""Local-registry proof rejects substitute sources and vacuous test output."""
import tempfile
import unittest
from pathlib import Path

import verify_llguidance_registry as audit


class RegistryTests(unittest.TestCase):
    def test_index_dependencies_preserve_registry_requirements_and_features(self):
        manifest = {"dependencies":{"anyhow":"1.0", "serde":{"version":"1.0","features":["derive"],"default-features":False}},"build-dependencies":{"cbindgen":{"version":"0.29","optional":True}},"target":{"cfg(unix)":{"dependencies":{"libc":"0.2"}}}}
        result = audit.registry_dependencies(manifest)
        self.assertEqual(len(result), 4)
        serde = next(dependency for dependency in result if dependency["name"] == "serde")
        self.assertEqual(serde["features"], ["derive"])
        self.assertFalse(serde["default_features"])
        self.assertEqual(serde["registry"], audit.CRATES_IO)
        self.assertEqual(next(dependency for dependency in result if dependency["name"] == "libc")["target"], "cfg(unix)")
        for source in [{"path":"../bad","version":"1"},{"git":"https://example.invalid/source","version":"1"}]:
            with self.assertRaisesRegex(ValueError, "published crates.io"):
                audit.registry_dependencies({"dependencies":{"bad":source}})

    def test_graph_requires_the_exact_registry_package_and_no_path_bypass(self):
        index = "sparse+http://127.0.0.1:12345/index/"
        package = {"id":"fork", "name":audit.PACKAGE, "version":"0.1.0-dev.6", "source":index,"manifest_path":"/cargo/registry/src/index/package/Cargo.toml"}
        metadata = {"resolve":{"nodes":[{"id":"fork"}]},"packages":[package]}
        audit.verify_registry_sources(metadata, index, "0.1.0-dev.6", Path("/cargo"))
        for source in [None, "git+https://example.invalid/fork", "registry+"+audit.CRATES_IO, "registry+"+index]:
            package["source"] = source
            with self.assertRaisesRegex(ValueError, "exact maintained compiler"):
                audit.verify_registry_sources(metadata, index, "0.1.0-dev.6", Path("/cargo"))

    def test_registry_source_identity_alone_cannot_hide_directory_replacement(self):
        index = "sparse+http://127.0.0.1:12345/index/"
        package = {"id":"fork", "name":audit.PACKAGE, "version":"0.1.0-dev.6", "source":index,"manifest_path":"/outside/source/Cargo.toml"}
        metadata = {"resolve":{"nodes":[{"id":"fork"}]},"packages":[package]}
        with self.assertRaisesRegex(ValueError, "directory/source replacement"):
            audit.verify_registry_sources(metadata, index, "0.1.0-dev.6", Path("/cargo"))

    def test_directory_substitution_and_cache_content_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive"
            archive.mkdir()
            (archive / "Cargo.toml").write_text("audited manifest")
            cached = root / "home" / "registry" / "src" / "index" / "package"
            cached.mkdir(parents=True)
            (cached / "Cargo.toml").write_text("audited manifest")
            package = {"manifest_path": str(cached / "Cargo.toml")}
            audit.verify_cached_contents(package, root / "home", archive)
            (cached / "Cargo.toml").write_text("substituted")
            with self.assertRaisesRegex(ValueError, "differs from the audited archive"):
                audit.verify_cached_contents(package, root / "home", archive)
            package["manifest_path"] = str(archive / "Cargo.toml")
            with self.assertRaisesRegex(ValueError, "outside Cargo's registry cache"):
                audit.verify_cached_contents(package, root / "home", archive)

    def test_positive_counts_and_both_patch_regressions_are_required(self):
        names = ["public_unsatisfiability_remains_distinct_from_invalid_schema_errors", "registry_compiler_preserves_known_key_lexeme_ownership"]
        output = "".join(f"test {name} ... ok\n" for name in names) + "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n"
        audit.positive_tests(output)
        for changed in [output.replace("2 passed", "0 passed"), output.replace(names[1], "wrong_test"), output + output]:
            with self.assertRaises(ValueError):
                audit.positive_tests(changed)


if __name__ == "__main__":
    unittest.main()
