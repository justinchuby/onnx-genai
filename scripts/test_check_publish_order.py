#!/usr/bin/env python3

import unittest
import tempfile
import hashlib
import json
from pathlib import Path

import check_publish_order


def package(
    name: str,
    manifest_path: str,
    *,
    publish=None,
    dependencies: list[tuple[str, str | None]] | None = None,
) -> dict:
    return {
        "id": f"path+file://{manifest_path}#{name}@0.1.0",
        "name": name,
        "manifest_path": manifest_path,
        "publish": publish,
        "dependencies": [
            {
                "name": dependency,
                "path": f"/repo/{dependency}",
                "kind": kind,
            }
            for dependency, kind in dependencies or []
        ],
    }


def metadata(packages: list[dict]):
    return {
        "packages": packages,
        "workspace_members": [entry["id"] for entry in packages],
        "workspace_root": "/repo",
    }


class PublishOrderTests(unittest.TestCase):
    def test_audited_external_vendor_is_not_a_missing_workspace_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "vendor" / "external"
            directory.mkdir(parents=True)
            manifest = b'[package]\nname = "external"\nversion = "1.9.1"\n'
            (directory / "Cargo.toml").write_bytes(manifest)
            record = {
                "crate": "external", "version": "1.9.1",
                "source": "https://crates.io/api/v1/crates/external/1.9.1/download",
                "sha256": "0" * 64,
                "upstream_file_sha256": {"Cargo.toml": hashlib.sha256(manifest).hexdigest()},
            }
            (directory / "UPSTREAM.json").write_text(json.dumps(record))
            owned = package("consumer", str(root / "consumer" / "Cargo.toml"))
            owned["dependencies"] = [{"name":"external","path":str(directory),"req":"^1.9.1"}]
            graph = metadata([owned])
            graph["workspace_root"] = str(root)
            graph["metadata"] = {"vendored-crates":["vendor/external/UPSTREAM.json"]}
            errors, edges = check_publish_order.validate_publish_order(["consumer"], graph, exclusions=frozenset())
            self.assertEqual(errors, [])
            self.assertEqual(edges, 0)
            owned["dependencies"][0]["req"] = "^2.0.0"
            errors, _ = check_publish_order.validate_publish_order(["consumer"], graph, exclusions=frozenset())
            self.assertTrue(any("identity/version differs" in error for error in errors))
            owned["dependencies"][0]["req"] = "^1.9.1"
            (directory / "Cargo.toml").write_bytes(manifest + b"# unaudited edit\n")
            errors, _ = check_publish_order.validate_publish_order(["consumer"], graph, exclusions=frozenset())
            self.assertTrue(any("differs from its published snapshot" in error for error in errors))
            (directory / "Cargo.toml").write_bytes(manifest)
            graph["metadata"] = {}
            errors, _ = check_publish_order.validate_publish_order(["consumer"], graph, exclusions=frozenset())
            self.assertTrue(any("not a workspace package" in error for error in errors))

    def test_owned_or_outside_sources_cannot_use_vendor_exemption(self):
        owned = package("consumer", "/repo/consumer/Cargo.toml")
        graph = metadata([owned])
        for declared, message in [
            (["consumer/UPSTREAM.json"], "workspace-owned"),
            (["../outside/UPSTREAM.json"], "escapes the workspace"),
        ]:
            graph["metadata"] = {"vendored-crates": declared}
            errors, _ = check_publish_order.validate_publish_order(["consumer"], graph, exclusions=frozenset())
            self.assertTrue(any(message in error for error in errors), errors)

    def test_nested_manifest_dependency_violation_is_detected(self):
        packages = [
            package(
                "onnx-genai-ort-sys",
                "/repo/crates/onnx-genai-ort/ort-sys/Cargo.toml",
                dependencies=[("onnx-genai-runtime-config", None)],
            ),
            package(
                "onnx-genai-runtime-config",
                "/repo/onnx-genai-runtime-config/Cargo.toml",
            ),
        ]
        graph = metadata(packages)

        problems, edges = check_publish_order.validate_publish_order(
            ["onnx-genai-ort-sys", "onnx-genai-runtime-config"],
            graph,
            frozenset(),
        )

        self.assertEqual(edges, 1)
        self.assertTrue(
            any("onnx-genai-runtime-config (#2)" in problem for problem in problems),
            problems,
        )

    def test_nested_manifest_dependency_first_order_is_valid(self):
        packages = [
            package("runtime-config", "/repo/crates/runtime-config/Cargo.toml"),
            package(
                "nested-sys",
                "/repo/crates/parent/sys/Cargo.toml",
                dependencies=[("runtime-config", None)],
            ),
        ]
        packages[0]["manifest_path"] = "/repo/runtime-config/Cargo.toml"
        graph = metadata(packages)

        problems, edges = check_publish_order.validate_publish_order(
            ["runtime-config", "nested-sys"], graph, frozenset()
        )

        self.assertEqual(problems, [])
        self.assertEqual(edges, 1)

    def test_duplicate_publish_entry_fails_closed(self):
        packages = [package("runtime-config", "/repo/crates/runtime-config/Cargo.toml")]
        graph = metadata(packages)

        problems, _ = check_publish_order.validate_publish_order(
            ["runtime-config", "runtime-config"], graph, frozenset()
        )

        self.assertIn("duplicate publish entries: runtime-config", problems)

    def test_dependency_omission_fails_closed(self):
        packages = [
            package("runtime-config", "/repo/runtime-config/Cargo.toml"),
            package(
                "nested-sys",
                "/repo/crates/parent/sys/Cargo.toml",
                dependencies=[("runtime-config", None)],
            ),
        ]
        graph = metadata(packages)

        problems, edges = check_publish_order.validate_publish_order(
            ["nested-sys"], graph, frozenset({"runtime-config"})
        )

        self.assertEqual(edges, 1)
        self.assertTrue(
            any("runtime-config, which is omitted" in problem for problem in problems),
            problems,
        )

    def test_dev_dependency_must_be_in_publish_order(self):
        packages = [
            package(
                "nested-sys",
                "/repo/crates/parent/sys/Cargo.toml",
                dependencies=[("test-helper", "dev")],
            ),
            package("test-helper", "/repo/test-helper/Cargo.toml", publish=[]),
        ]
        graph = metadata(packages)

        problems, edges = check_publish_order.validate_publish_order(
            ["nested-sys"], graph, frozenset()
        )

        self.assertEqual(edges, 1)
        self.assertTrue(
            any("test-helper, which is omitted" in problem for problem in problems),
            problems,
        )

    def test_dev_dependency_must_be_published_first(self):
        packages = [
            package(
                "consumer",
                "/repo/consumer/Cargo.toml",
                dependencies=[("test-helper", "dev")],
            ),
            package("test-helper", "/repo/test-helper/Cargo.toml"),
        ]
        graph = metadata(packages)

        problems, edges = check_publish_order.validate_publish_order(
            ["consumer", "test-helper"], graph, frozenset()
        )

        self.assertEqual(edges, 1)
        self.assertTrue(
            any("test-helper (#2), which is published later" in p for p in problems),
            problems,
        )

        problems, edges = check_publish_order.validate_publish_order(
            ["test-helper", "consumer"], graph, frozenset()
        )

        self.assertEqual(problems, [])
        self.assertEqual(edges, 1)

    def test_duplicate_workspace_package_name_fails_closed(self):
        packages = [
            package("same-name", "/repo/first/Cargo.toml"),
            package("same-name", "/repo/second/Cargo.toml"),
        ]

        problems, _ = check_publish_order.validate_publish_order(
            ["same-name"], metadata(packages), frozenset()
        )

        self.assertTrue(
            any("workspace package names are not unique" in problem for problem in problems),
            problems,
        )

    def test_missing_workspace_package_entry_fails_closed(self):
        packages = [package("runtime-config", "/repo/runtime-config/Cargo.toml")]
        graph = metadata(packages)
        graph["workspace_members"].append("path+file:///repo/missing#missing@0.1.0")

        problems, _ = check_publish_order.validate_publish_order(
            ["runtime-config"], graph, frozenset()
        )

        self.assertTrue(
            any("workspace members missing package entries" in problem for problem in problems),
            problems,
        )


if __name__ == "__main__":
    unittest.main()
