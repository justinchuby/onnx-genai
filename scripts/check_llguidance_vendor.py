#!/usr/bin/env python3
"""Audit the single maintained compiler tree and its upstream/archive provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tomllib

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "llguidance"
GENERATED = {"Cargo.toml.orig", "Cargo.lock", ".cargo_vcs_info.json"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError(f"compiler source {name!r} is missing/outside its package; restore the audited file")
    return path


def workspace_version(root: Path) -> str:
    for directory in root.parents:
        manifest = directory / "Cargo.toml"
        if manifest.is_file():
            version = tomllib.loads(manifest.read_text()).get("workspace", {}).get("package", {}).get("version")
            if isinstance(version, str):
                return version
    raise RuntimeError("managed compiler version inherits a missing workspace; audit a package archive with --packaged")


def verify(root: Path, *, packaged: bool = False) -> dict:
    provenance = json.loads(source_file(root, "UPSTREAM.json").read_text())
    original = provenance["upstream_file_sha256"]
    if provenance["source"] != f"https://crates.io/api/v1/crates/{provenance['crate']}/{provenance['version']}/download":
        raise RuntimeError("compiler provenance must identify the original published archive")
    if len(provenance["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in provenance["sha256"]):
        raise RuntimeError("original compiler archive checksum is invalid")
    locations = provenance["upstream_file_locations"]
    expected_changes = set(provenance["modified_files"])
    omitted = set(provenance.get("omitted_files", {}))
    if not omitted <= set(original) or not set(locations) <= set(original):
        raise RuntimeError("compiler omissions/relocations must identify original archive files")
    if len(set(locations.values())) != len(locations) or set(locations) & (omitted | expected_changes):
        raise RuntimeError("preserved upstream metadata must have unique immutable locations")
    actual_changes = set()
    allowed = {"UPSTREAM.json", "NXRT-PATCHES.md"}
    for name, expected in original.items():
        if name in omitted:
            if (root / name).exists() and not (packaged and name in GENERATED):
                raise RuntimeError(f"reserved original file {name} must remain omitted from maintained sources")
            continue
        relative = locations.get(name, name)
        allowed.add(relative)
        path = source_file(root, relative)
        if digest(path) != expected:
            if name in locations:
                raise RuntimeError(f"preserved upstream metadata {relative} changed; restore its original archive bytes")
            actual_changes.add(name)
    if actual_changes != expected_changes:
        raise RuntimeError(
            f"compiler patch footprint differs: expected {sorted(expected_changes)}, found {sorted(actual_changes)}; audit the intended source patch"
        )
    patched = provenance["patched_file_sha256"]
    if set(patched) != expected_changes:
        raise RuntimeError("compiler patched hashes must cover exactly the declared patch footprint")
    for name, expected in patched.items():
        if digest(source_file(root, name)) != expected:
            raise RuntimeError(f"compiler patch hash changed for {name}; audit and record the intended patch")
    maintained = provenance["maintained_file_sha256"]
    if "Cargo.toml" not in maintained:
        raise RuntimeError("maintained compiler manifest needs its own audited source hash")
    for name, expected in maintained.items():
        relative = "Cargo.toml.orig" if packaged and name == "Cargo.toml" else name
        allowed.add(relative)
        if digest(source_file(root, relative)) != expected:
            raise RuntimeError(f"maintained compiler file {relative} changed; audit and record its intended contents")
    manifest = tomllib.loads(source_file(root, "Cargo.toml").read_text())
    identity = provenance["maintained_package"]
    actual = manifest["package"]
    version = actual["version"]
    if isinstance(version, dict) and version == {"workspace": True} and not packaged:
        version = workspace_version(root)
    if (actual["name"], version, actual["edition"]) != (identity["name"], identity["version"], identity["edition"]):
        raise RuntimeError("maintained compiler package identity/version/edition differs from its audited declaration")
    if actual.get("license") != "MIT":
        raise RuntimeError("maintained compiler must retain its upstream MIT license")
    if packaged:
        allowed |= GENERATED | {"Cargo.toml"}
        source_manifest = tomllib.loads(source_file(root, "Cargo.toml.orig").read_text())
        if manifest.get("features", {}) != source_manifest.get("features", {}):
            raise RuntimeError("packaged compiler feature semantics differ from its audited source manifest")
    extra = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - allowed
    if extra:
        raise RuntimeError(f"unexpected compiler files {sorted(extra)}; remove generated files or audit their provenance")
    return {
        "package": identity["name"], "version": identity["version"],
        "upstream": f"{provenance['crate']} {provenance['version']}",
        "original_records": len(original), "omissions": len(omitted),
        "source_patches": len(actual_changes), "packaged": packaged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--packaged", action="store_true", help="audit an extracted cargo package and its Cargo-generated metadata")
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve(), packaged=args.packaged)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
