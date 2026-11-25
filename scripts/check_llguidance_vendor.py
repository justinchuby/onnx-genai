#!/usr/bin/env python3
"""Verify the pinned compiler source and the declared local patch footprint."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "vendor" / "llguidance"
    provenance = json.loads((root / "UPSTREAM.json").read_text())
    original = provenance["upstream_file_sha256"]
    expected_changes = set(provenance["modified_files"])
    omitted = set(provenance.get("omitted_files", {}))
    if not omitted <= set(original):
        raise RuntimeError("vendored omissions must name original archive files")
    actual_changes = set()
    for name, expected in original.items():
        path = root / name
        if name in omitted:
            if path.exists():
                raise RuntimeError(f"reserved vendored file {name} must remain omitted for Cargo packaging")
            continue
        if not path.is_file():
            raise RuntimeError(f"vendored llguidance source is missing {name}; restore it from the pinned crate archive")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            actual_changes.add(name)
    if actual_changes != expected_changes:
        raise RuntimeError(
            f"vendored llguidance patch footprint differs: expected {sorted(expected_changes)}, "
            f"found {sorted(actual_changes)}; restore upstream files or explicitly audit and record the patch update"
        )
    patched = provenance["patched_file_sha256"]
    if set(patched) != expected_changes:
        raise RuntimeError("vendored llguidance patched hashes do not match declared files; record every audited patch")
    for name, expected in patched.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"vendored llguidance patch hash changed for {name}; audit and record the intended patch before updating its hash")
    allowed = set(original) | {"UPSTREAM.json", "NXRT-PATCHES.md"}
    extra = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - allowed
    if extra:
        raise RuntimeError(f"unexpected vendored llguidance files {sorted(extra)}; remove generated artifacts or record their provenance")
    print(f"Verified llguidance {provenance['version']}: {len(original)} original records, {len(omitted)} recorded omission, {len(actual_changes)} declared compiler patch")


if __name__ == "__main__":
    main()
