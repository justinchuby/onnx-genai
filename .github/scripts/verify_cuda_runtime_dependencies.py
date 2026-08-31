#!/usr/bin/env python3
"""Verify every Python CUDA surface follows the repository dependency lock."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements-cuda-dev.txt"
PLUGIN = ROOT / "python" / "nxrt-ep-cuda" / "pyproject.toml"
NXRT = ROOT / "crates" / "onnx-runtime-python" / "pyproject.toml"
EXACT = re.compile(r"^(?P<name>nvidia-[A-Za-z0-9-]+)==(?P<version>[A-Za-z0-9_.+-]+)$")


def exact_nvidia_requirements(requirements: list[str], source: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    for requirement in requirements:
        requirement = requirement.split("#", 1)[0].strip()
        if not requirement or not requirement.lower().startswith("nvidia-"):
            continue
        match = EXACT.fullmatch(requirement)
        if match is None:
            raise ValueError(
                f"{source.relative_to(ROOT)} contains non-exact NVIDIA requirement "
                f"{requirement!r}; use package==version"
            )
        name = match.group("name").lower()
        if name in locked:
            raise ValueError(f"{source.relative_to(ROOT)} repeats {name}")
        locked[name] = match.group("version")
    if not locked:
        raise ValueError(f"{source.relative_to(ROOT)} contains no NVIDIA requirements")
    return locked


def load_lock() -> dict[str, str]:
    return exact_nvidia_requirements(LOCK.read_text(encoding="utf-8").splitlines(), LOCK)


def load_pyproject(path: Path, table: tuple[str, ...]) -> dict[str, str]:
    value: object = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in table:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path.relative_to(ROOT)} is missing {'.'.join(table)}")
        value = value[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path.relative_to(ROOT)} {'.'.join(table)} must be a string list")
    return exact_nvidia_requirements(value, path)


def verify_local() -> dict[str, str]:
    locked = load_lock()
    surfaces = {
        PLUGIN: load_pyproject(PLUGIN, ("project", "dependencies")),
        NXRT: load_pyproject(NXRT, ("project", "optional-dependencies", "cuda")),
    }
    errors = [
        f"{path.relative_to(ROOT)} CUDA dependencies differ from "
        f"{LOCK.relative_to(ROOT)}:\n  expected={locked}\n  actual={actual}"
        for path, actual in surfaces.items()
        if actual != locked
    ]
    if errors:
        raise ValueError("\n".join(errors))
    return locked


def verify_pypi(locked: dict[str, str]) -> None:
    for name, version in sorted(locked.items()):
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot verify {name}=={version} at {url}: {error}") from error
        files = payload.get("urls", [])
        wheels = [
            file
            for file in files
            if file.get("packagetype") == "bdist_wheel" and not file.get("yanked", False)
        ]
        if not wheels:
            raise ValueError(f"{name}=={version} has no non-yanked wheel at {url}")
        print(f"{name}=={version}: {len(wheels)} live non-yanked wheel(s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pypi",
        action="store_true",
        help="also require each exact release to have a live non-yanked PyPI wheel",
    )
    args = parser.parse_args()
    try:
        locked = verify_local()
        print(
            f"CUDA dependency lock matches {PLUGIN.relative_to(ROOT)} and "
            f"{NXRT.relative_to(ROOT)} ({len(locked)} exact packages)"
        )
        if args.pypi:
            verify_pypi(locked)
    except ValueError as error:
        print(f"CUDA dependency verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
