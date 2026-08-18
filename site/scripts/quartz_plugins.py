#!/usr/bin/env python3
"""Prepare and verify Quartz plugins against quartz.lock.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def enabled_plugin_names(config: Path) -> set[str]:
    names: set[str] = set()
    source: str | None = None
    enabled = False
    for line in config.read_text(encoding="utf-8").splitlines():
        source_match = re.match(r"^\s{2}- source:\s*(\S+)\s*$", line)
        if source_match:
            if source and enabled:
                names.add(source.rsplit("/", 1)[-1].split("#", 1)[0])
            source = source_match.group(1)
            enabled = False
            continue
        enabled_match = re.match(r"^\s{4}enabled:\s*(true|false)\s*$", line)
        if enabled_match and source:
            enabled = enabled_match.group(1) == "true"
    if source and enabled:
        names.add(source.rsplit("/", 1)[-1].split("#", 1)[0])
    return names


def read_lock(lock_path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        raise ValueError(f"{lock_path} has no plugins object")
    return plugins


def plugin_head(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def validate_inputs(
    config: Path, lock_path: Path, plugins_dir: Path
) -> tuple[dict[str, dict[str, str]], Path]:
    lock = read_lock(lock_path)
    enabled = enabled_plugin_names(config)
    missing = sorted(enabled - set(lock))
    if missing:
        raise ValueError(f"enabled plugins missing from {lock_path}: {', '.join(missing)}")
    root = plugins_dir.resolve()
    for name, entry in lock.items():
        if not PLUGIN_NAME.fullmatch(name):
            raise ValueError(f"unsafe plugin name in {lock_path}: {name!r}")
        commit = entry.get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"plugin {name} is not pinned to a 40-character commit")
    return lock, root


def prepare(lock: dict[str, dict[str, str]], plugins_dir: Path) -> None:
    plugins_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    clean_ci = os.environ.get("CI", "").casefold() == "true"
    for name, entry in lock.items():
        directory = plugins_dir / name
        expected = entry["commit"]
        if directory.exists() and (
            clean_ci
            or plugin_head(directory) != expected
            or not (directory / "dist" / "index.js").is_file()
        ):
            shutil.rmtree(directory)
            removed += 1
    print(f"Prepared pinned plugin cache ({removed} stale plugin(s) removed).")


def verify(lock: dict[str, dict[str, str]], plugins_dir: Path) -> int:
    errors: list[str] = []
    for name, entry in lock.items():
        directory = plugins_dir / name
        actual = plugin_head(directory)
        if actual != entry["commit"]:
            errors.append(f"{name}: expected {entry['commit']}, found {actual or 'missing'}")
        if not (directory / "dist" / "index.js").is_file():
            errors.append(f"{name}: missing built dist/index.js")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(lock)} Quartz plugin commit(s) and build output(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("config", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("plugins", type=Path)
    args = parser.parse_args()
    try:
        lock, plugins_dir = validate_inputs(args.config, args.lock, args.plugins)
        if args.command == "prepare":
            prepare(lock, plugins_dir)
            return 0
        return verify(lock, plugins_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
