#!/usr/bin/env python3
"""Verify an exact maintained compiler .crate through a temporary local registry.

Run under the shared host workload lock. No package is published. Lockfile
creation may fetch registry metadata/dependencies; the consumer tests then run
locked and offline. Artifacts/logs remain in the requested external directory.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

import tomllib
from audit_llama_server_conformance import cargo_context, run_contained
from check_llguidance_vendor import DEFAULT_ROOT, digest, verify

CRATES_IO = "https://github.com/rust-lang/crates.io-index"
PACKAGE = "onnx-genai-llguidance"


def extract_archive(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if sum(member.size for member in members) > 64 * 1024 * 1024:
            raise ValueError("compiler archive exceeds the 64 MiB audit budget")
        roots = set()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsafe compiler archive entry {member.name!r}; use the reviewed Cargo package")
            roots.add(relative.parts[0])
        if len(roots) != 1:
            raise ValueError("compiler archive must contain exactly one Cargo package directory")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_file, target.open("wb") as output:
                    shutil.copyfileobj(input_file, output)
    return destination / roots.pop()


def registry_dependencies(manifest: dict) -> list[dict]:
    result = []

    def group(table: dict, target: str | None = None) -> None:
        for key, kind in [("dependencies", "normal"), ("dev-dependencies", "dev"), ("build-dependencies", "build")]:
            for name, declaration in table.get(key, {}).items():
                dependency = {"version": declaration} if isinstance(declaration, str) else declaration
                if "version" not in dependency or any(field in dependency for field in ["path", "git", "registry", "registry-index"]):
                    raise ValueError(f"normalized compiler dependency {name} must use its published crates.io requirement")
                result.append({
                    "name": name, "req": dependency["version"],
                    "features": dependency.get("features", []),
                    "optional": dependency.get("optional", False),
                    "default_features": dependency.get("default-features", True),
                    "target": target, "kind": kind, "registry": CRATES_IO,
                    "package": dependency.get("package"),
                })

    group(manifest)
    for target, table in manifest.get("target", {}).items():
        group(table, target)
    return result


def verify_registry_sources(metadata: dict, index: str, version: str, cargo_home: Path) -> dict:
    selected = {node["id"] for node in metadata["resolve"]["nodes"]}
    packages = [package for package in metadata["packages"] if package["id"] in selected]
    compiler = [package for package in packages if package["name"] == PACKAGE]
    if len(compiler) != 1 or compiler[0]["version"] != version or compiler[0]["source"] != index:
        raise ValueError("consumer did not resolve the exact maintained compiler from the temporary registry; remove path/git/patch overrides")
    for package in packages:
        if package["id"] == metadata["resolve"].get("root"):
            continue
        if package["name"] == "llguidance" or not (package["source"] or "").startswith(("registry+", "sparse+")):
            raise ValueError(f"consumer includes nonregistry/unpatched package {package['name']}; no path or patch bypass is permitted")
        if not Path(package["manifest_path"]).parent.resolve().is_relative_to((cargo_home / "registry" / "src").resolve()):
            raise ValueError(f"consumer package {package['name']} uses a directory/source replacement outside Cargo's registry cache")
    return compiler[0]


def verify_cached_contents(package: dict, cargo_home: Path, extracted: Path) -> None:
    cached = Path(package["manifest_path"]).parent.resolve()
    if not cached.is_relative_to((cargo_home / "registry" / "src").resolve()):
        raise ValueError("compiler source is outside Cargo's registry cache; directory/path source replacement is not a registry artifact proof")
    for expected in extracted.rglob("*"):
        if expected.is_file():
            actual = cached / expected.relative_to(extracted)
            if not actual.is_file() or digest(actual) != digest(expected):
                raise ValueError(f"cached registry compiler differs from the audited archive at {expected.relative_to(extracted)}")


def positive_tests(output: str) -> None:
    expected = [
        "public_unsatisfiability_remains_distinct_from_invalid_schema_errors",
        "registry_compiler_preserves_known_key_lexeme_ownership",
    ]
    if re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", output) != [("2", "0", "0")]:
        raise ValueError("registry consumer requires exactly two passed, zero failed/ignored tests; inspect consumer-test.log")
    for name in expected:
        if not re.search(r"^test " + re.escape(name) + r" \.\.\. ok$", output, re.MULTILINE):
            raise ValueError(f"registry consumer did not execute required test {name}")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


def run(args: argparse.Namespace) -> None:
    archive = args.archive.resolve()
    root = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if root.is_relative_to(repository):
        raise ValueError("registry verification artifacts must be outside the repository")
    root.mkdir(parents=True, exist_ok=False)
    original_hash = digest(archive)
    extracted = extract_archive(archive, root / "extracted")
    if (extracted / "UPSTREAM.json").read_bytes() != (DEFAULT_ROOT / "UPSTREAM.json").read_bytes():
        raise ValueError("archive provenance differs from the reviewed canonical compiler source")
    audited = verify(extracted, packaged=True)
    manifest = tomllib.loads((extracted / "Cargo.toml").read_text())
    version = audited["version"]
    registry = root / "registry"
    registry.mkdir()
    handler = functools.partial(QuietHandler, directory=str(registry))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    base = f"http://127.0.0.1:{server.server_port}"
    index = "sparse+" + base + "/index/"
    record = {
        "name": PACKAGE, "vers": version, "deps": registry_dependencies(manifest),
        "cksum": original_hash, "features": {}, "features2": manifest.get("features", {}),
        "v": 2, "yanked": False, "links": manifest["package"].get("links"),
        "rust_version": manifest["package"].get("rust-version"),
    }
    index_file = registry / "index" / PACKAGE[:2] / PACKAGE[2:4] / PACKAGE
    index_file.parent.mkdir(parents=True)
    index_file.write_text(json.dumps(record) + "\n")
    (registry / "index" / "config.json").write_text(json.dumps({"dl":base + "/crates/{crate}/{version}/download"}))
    download = registry / "crates" / PACKAGE / version / "download"
    download.parent.mkdir(parents=True)
    shutil.copyfile(archive, download)
    consumer = root / "consumer"
    (consumer / "tests").mkdir(parents=True)
    (consumer / "src").mkdir()
    (consumer / "src" / "lib.rs").write_text("// Registry-only consumer; behavioral tests live in tests/registry_surface.rs.\n")
    (consumer / ".cargo").mkdir()
    (consumer / "Cargo.toml").write_text(
        '[package]\nname="nxrt-registry-consumer"\nversion="0.0.0"\nedition="2021"\n'
        '[dependencies]\nonnx-genai-llguidance={version="=' + version + '", registry="nxrt-audit"}\n'
        '[dev-dependencies]\nserde_json="1.0.138"\n[workspace]\n'
    )
    (consumer / ".cargo" / "config.toml").write_text('[registries.nxrt-audit]\nindex=' + json.dumps(index) + '\n')
    shutil.copyfile(extracted / "tests" / "registry_surface.rs", consumer / "tests" / "registry_surface.rs")
    environment = dict(os.environ)
    context = cargo_context(consumer, args.cargo_config, environment)
    command = ["cargo"]
    if args.cargo_config:
        command += ["--config", context["groups_low_to_high"][-1]["selected"]]
    receipt = {"status":"running", "archive":str(archive), "archive_sha256":original_hash, "package_audit":audited, "registry_index":index, "cargo_context":context, "commands":[]}
    receipt_file = root / "receipt.json"

    def save() -> None:
        temporary = receipt_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(receipt_file)

    def cargo(phase: str, arguments: list[str]) -> Path:
        invocation = [*command, *arguments]
        log = root / (phase + ".log")
        receipt["commands"].append({"phase":phase,"argv":invocation,"log":str(log)})
        save()
        with log.open("w") as output:
            completed = run_contained(invocation, cwd=consumer, stdout=output, timeout=args.timeout_seconds, env=environment)
        if completed.returncode:
            raise ValueError(f"{phase} exited {completed.returncode}; inspect {log}")
        if cargo_context(consumer, args.cargo_config, dict(os.environ)) != context or digest(archive) != original_hash:
            raise ValueError("registry verification configuration/archive changed during execution")
        return log

    save()
    server_thread.start()
    try:
        cargo("consumer-lock", ["generate-lockfile"])
        lock_hash = digest(consumer / "Cargo.lock")
        metadata_log = cargo("consumer-metadata", ["metadata","--locked","--format-version","1"])
        metadata_output = metadata_log.read_text()
        metadata_lines = [line for line in metadata_output.splitlines() if line.startswith("{")]
        if len(metadata_lines) != 1:
            raise ValueError("cargo metadata did not emit one package graph; inspect consumer-metadata.log")
        metadata = json.loads(metadata_lines[0])
        selected = verify_registry_sources(metadata, index, version, Path(context["cargo_home"]))
        verify_cached_contents(selected, Path(context["cargo_home"]), extracted)
        test_log = cargo("consumer-test", ["test","--locked","--offline","-j",str(args.jobs),"--test","registry_surface","--","--test-threads=1"])
        positive_tests(test_log.read_text())
        verify_cached_contents(selected, Path(context["cargo_home"]), extracted)
        if digest(consumer / "Cargo.lock") != lock_hash:
            raise ValueError("registry consumer lockfile changed after resolution")
        receipt.update(status="passed", consumer_lock_sha256=lock_hash, tests_passed=2, tests_failed=0, tests_ignored=0)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
        save()
    print(json.dumps({"status":receipt["status"],"receipt":str(receipt_file),"tests_passed":2}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cargo-config", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout_seconds <= 0 or not 0 <= args.port <= 65535:
        parser.error("jobs/timeout must be positive and port must be 0..65535")
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
