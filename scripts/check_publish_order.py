#!/usr/bin/env python3
"""Verify the crates.io release set and its dependency-first publish order.

`cargo publish` uploads one crate at a time and resolves each against
crates.io, so a crate listed before something it depends on fails outright --
the required version is not there yet. That is invisible until a dependency
edge changes: the tracer sat mid-list happily for months, then startup tracing
gave the loader a dependency on it and the next release died partway through,
leaving some crates published at the new version and some not.

Dev-dependencies are ignored: they are not resolved when packaging.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/publish.yml"

# These workspace packages permit publishing but are intentionally outside the
# crates.io release job. Packages with `publish = false` are excluded directly
# from Cargo metadata and do not need to be repeated here.
EXCLUDED_PUBLISHABLE_PACKAGES = frozenset(
    {
        "onnx-genai-capi",
        "onnx-genai-paged-attention",
        "onnx-runtime-cost-model",
        "onnx-runtime-ep-cpu-plugin",
        "onnx-runtime-ep-cuda-plugin",
        "onnx-runtime-ep-nxrt-abi",
        "onnx-runtime-ep-plugin",
        "onnx-runtime-hostmon",
        "onnx-runtime-memory-abi",
        "onnx-runtime-memory-host",
        "onnx-runtime-operator-selection",
    }
)
PUBLISH_LINE = re.compile(r"^\s*publish_crate\s+([A-Za-z0-9_-]+)\s*(?:#.*)?$")


def published_order(workflow: Path = WORKFLOW) -> list[str]:
    order = []
    for line in workflow.read_text().splitlines():
        if match := PUBLISH_LINE.match(line):
            order.append(match.group(1))
    return order


def cargo_metadata(root: Path = ROOT) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--format-version",
                "1",
                "--no-deps",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"cargo metadata failed: {detail.strip()}") from error
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"cargo metadata returned invalid JSON: {error}") from error


def _is_publishable(package: dict[str, Any]) -> bool:
    return package.get("publish") != []


def validate_publish_order(
    order: list[str],
    metadata: dict[str, Any],
    exclusions: frozenset[str] = EXCLUDED_PUBLISHABLE_PACKAGES,
) -> tuple[list[str], int]:
    problems = []
    duplicates = sorted(
        name for name, count in collections.Counter(order).items() if count > 1
    )
    if duplicates:
        problems.append(
            "duplicate publish entries: " + ", ".join(duplicates)
        )

    packages = metadata.get("packages")
    workspace_members = metadata.get("workspace_members")
    if not isinstance(packages, list) or not isinstance(workspace_members, list):
        return ["cargo metadata omitted packages or workspace_members"], 0

    packages_by_id = {}
    workspace_names: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    workspace_paths: dict[Path, list[dict[str, Any]]] = collections.defaultdict(list)
    workspace_ids = set(workspace_members)
    for package in packages:
        package_id = package.get("id")
        if not isinstance(package_id, str):
            problems.append("cargo metadata contains a package without an id")
            continue
        if package_id in packages_by_id:
            problems.append(f"cargo metadata repeats package id {package_id}")
        packages_by_id[package_id] = package
        if package_id in workspace_ids:
            name = package.get("name")
            if not isinstance(name, str):
                problems.append(f"workspace package {package_id} has no name")
            else:
                workspace_names[name].append(package)
            manifest_path = package.get("manifest_path")
            if not isinstance(manifest_path, str):
                problems.append(f"workspace package {package_id} has no manifest path")
            else:
                workspace_paths[Path(manifest_path).parent.resolve()].append(package)

    missing_ids = sorted(workspace_ids - packages_by_id.keys())
    if missing_ids:
        problems.append(
            "workspace members missing package entries: " + ", ".join(missing_ids)
        )
    duplicate_names = sorted(
        name for name, entries in workspace_names.items() if len(entries) != 1
    )
    if duplicate_names:
        problems.append(
            "workspace package names are not unique: " + ", ".join(duplicate_names)
        )
    duplicate_paths = sorted(
        str(path) for path, entries in workspace_paths.items() if len(entries) != 1
    )
    if duplicate_paths:
        problems.append(
            "workspace package manifest directories are not unique: "
            + ", ".join(duplicate_paths)
        )

    known_names = set(workspace_names)
    unknown = sorted(set(order) - known_names)
    if unknown:
        problems.append(
            "publish entries missing from the Cargo workspace: " + ", ".join(unknown)
        )

    stale_exclusions = sorted(exclusions - known_names)
    if stale_exclusions:
        problems.append(
            "publish exclusions missing from the Cargo workspace: "
            + ", ".join(stale_exclusions)
        )

    intended = {
        name
        for name, entries in workspace_names.items()
        if len(entries) == 1
        and _is_publishable(entries[0])
        and name not in exclusions
    }
    missing = sorted(intended - set(order))
    if missing:
        problems.append(
            "publishable workspace crates omitted from the workflow: "
            + ", ".join(missing)
        )
    unexpected = sorted(set(order) - intended)
    if unexpected:
        problems.append(
            "workflow includes excluded or non-publishable crates: "
            + ", ".join(unexpected)
        )

    position = {crate: index for index, crate in enumerate(order)}
    edge_count = 0
    workspace_root = Path(metadata.get("workspace_root", ROOT)).resolve()
    for crate in order:
        entries = workspace_names.get(crate, [])
        if len(entries) != 1:
            continue
        package = entries[0]
        dependencies = package.get("dependencies")
        if not isinstance(dependencies, list):
            problems.append(f"{crate} has no dependency list in Cargo metadata")
            continue
        for dependency in dependencies:
            if dependency.get("kind") == "dev":
                continue
            dependency_path = dependency.get("path")
            if not isinstance(dependency_path, str):
                continue
            resolved_path = Path(dependency_path).resolve()
            dependency_entries = workspace_paths.get(resolved_path, [])
            if not dependency_entries:
                if resolved_path.is_relative_to(workspace_root):
                    problems.append(
                        f"{crate} has workspace-local path dependency "
                        f"{dependency.get('name', resolved_path.name)} at "
                        f"{resolved_path}, but it is not a workspace package"
                    )
                continue
            if len(dependency_entries) != 1:
                problems.append(
                    f"{crate} dependency path {resolved_path} resolves to "
                    "multiple workspace packages"
                )
                continue
            dependency_name = dependency_entries[0]["name"]
            edge_count += 1
            if dependency_name not in position:
                problems.append(
                    f"{crate} depends on {dependency_name}, which is omitted "
                    "from the publish workflow"
                )
            elif position[dependency_name] > position[crate]:
                problems.append(
                    f"{crate} (#{position[crate] + 1}) depends on "
                    f"{dependency_name} (#{position[dependency_name] + 1}), "
                    "which is published later"
                )
    return problems, edge_count


def main() -> int:
    order = published_order()
    if not order:
        print(f"No publish_crate lines found in {WORKFLOW}", file=sys.stderr)
        return 1
    try:
        metadata = cargo_metadata()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    problems, edge_count = validate_publish_order(order, metadata)
    if problems:
        print(
            "Publish workflow validation failed:\n  "
            + "\n  ".join(problems)
            + f"\n\nFix the release set or dependency order in {WORKFLOW.name}.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Publish order is complete and topologically valid for {len(order)} "
        f"crates ({edge_count} workspace dependency edges)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
