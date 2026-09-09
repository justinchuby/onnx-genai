#!/usr/bin/env python3

import unittest

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

    def test_dev_dependency_does_not_constrain_publish_order(self):
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

        self.assertEqual(problems, [])
        self.assertEqual(edges, 0)

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
