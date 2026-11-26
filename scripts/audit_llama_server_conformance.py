"""Inventory behavioral acceptance separately from static endpoint registration.

--check/--write inspect source links, not runtime behavior. --run resolves each
Rust test's exact identity, executes it and writes a revision-bound receipt.
Run cargo work under the host's shared workload lock; this script does not own it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
from pathlib import Path

import tomllib
from audit_llama_server_routes import EXTERNAL_HARNESS, contracts

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = Path("tests/fixtures/llama-server-conformance.json")
REPORT = Path("docs/LLAMA_CPP_BEHAVIOR_MATRIX.md")
ACCEPTANCE = {
    "frontend_cli",
    "discovery_templates_token_counts",
    "replay_cancellation_rollback",
    "completion_embedding_infill",
    "responses_messages_tools",
    "sampling_constraints_reporting",
    "ranking_lora_slots",
    "model_sources_audio",
    "full_acceptance_publication",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(root: Path, name: str) -> Path:
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(
            f"invalid evidence source {name!r}; supply an existing repository file"
        )
    return path


def package_target(root: Path, case: dict, identity: str) -> Path:
    package = case.get("package")
    manifest_name = case.get("package_manifest")
    if (
        not isinstance(package, str)
        or not re.fullmatch(r"[a-z][a-z0-9_-]+", package)
        or not isinstance(manifest_name, str)
    ):
        raise ValueError(
            f"{identity}: Rust evidence requires explicit package and package_manifest fields"
        )
    manifest = source_path(root, manifest_name)
    if manifest.name != "Cargo.toml":
        raise ValueError(f"{identity}: package_manifest must name a local Cargo.toml")
    package_root = manifest.parent.resolve()
    document = tomllib.loads(manifest.read_text())
    if document.get("package", {}).get("name") != package:
        raise ValueError(
            f"{identity}: package {package!r} does not match {manifest_name}"
        )
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]
    members = {
        path.resolve()
        for pattern in workspace["members"]
        for path in root.glob(pattern)
    }
    excluded = {
        path.resolve()
        for pattern in workspace.get("exclude", [])
        for path in root.glob(pattern)
    }
    if package_root not in members or package_root in excluded:
        raise ValueError(
            f"{identity}: {manifest_name} is not a selected workspace member; Cargo -p must select this actual package"
        )
    source = source_path(root, case["source"]).resolve()
    if not source.is_relative_to(package_root):
        raise ValueError(f"{identity}: evidence source is outside its declared package")
    target = case["target"]
    if target == "lib":
        library = document.get("lib", {})
        if library.get("harness") is False:
            raise ValueError(
                f"{identity}: custom lib test harness cannot provide the exact libtest evidence contract"
            )
        if library.get("test") is False or (
            "lib" not in document and document["package"].get("autolib") is False
        ):
            raise ValueError(f"{identity}: package has no enabled lib test target")
        entry = package_root / library.get("path", "src/lib.rs")
        if not entry.is_file() or not source.is_relative_to(entry.parent.resolve()):
            raise ValueError(
                f"{identity}: source does not belong to the declared lib test target directory"
            )
    elif re.fullmatch(r"test:[a-z0-9_]+", target):
        name = target.split(":", 1)[1]
        declared = [
            item for item in document.get("test", []) if item.get("name") == name
        ]
        if len(declared) > 1:
            raise ValueError(f"{identity}: duplicate integration test target {name}")
        if declared:
            if declared[0].get("harness") is False:
                raise ValueError(
                    f"{identity}: custom test harness cannot provide the exact libtest evidence contract"
                )
            entries = [package_root / declared[0].get("path", f"tests/{name}.rs")]
        elif document["package"].get("autotests", True):
            entries = [
                path
                for path in [
                    package_root / f"tests/{name}.rs",
                    package_root / f"tests/{name}/main.rs",
                ]
                if path.is_file()
            ]
        else:
            entries = []
        if (
            len(entries) != 1
            or not entries[0].is_file()
            or source != entries[0].resolve()
        ):
            raise ValueError(
                f"{identity}: evidence must name the existing declared/automatic integration test entrypoint for {name}"
            )
    else:
        raise ValueError(f"{identity}: invalid Rust target; use lib or test:NAME")
    return manifest


def validate(root: Path, inventory: dict) -> dict:
    fixture = json.loads((root / "tests/fixtures/llama-server-routes.json").read_text())
    if (
        inventory.get("version") != 1
        or inventory.get("upstream_commit") != fixture["commit"]
    ):
        raise ValueError(
            "conformance inventory version/revision differs from pinned routes; review both deliberately"
        )
    rows = contracts(
        fixture, (root / "crates/onnx-genai-server/src/lib.rs").read_text()
    )
    evidence = inventory["evidence"]
    if not evidence:
        raise ValueError(
            "empty behavioral evidence inventory; link concrete acceptance tests"
        )
    for identity, case in evidence.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]+", identity):
            raise ValueError(
                f"invalid evidence ID {identity!r}; use lowercase identifier characters"
            )
        source = source_path(root, case["source"]).read_text()
        if not case.get("assertions") or not case.get("execution_contract"):
            raise ValueError(
                f"{identity}: describe assertions and the execution contract"
            )
        if case["kind"] == "rust":
            package_target(root, case, identity)
            leaf = case["test"].split("::")[-1]
            functions = list(
                re.finditer(
                    r"\b(?:async\s+)?fn\s+" + re.escape(leaf) + r"\s*\(", source
                )
            )
            if len(functions) != 1:
                raise ValueError(
                    f"{identity}: exact source function {leaf!r} missing/ambiguous; update the evidence link"
                )
            attributes = re.search(
                r"(?:#\[[^\]]*\]\s*)+$", source[: functions[0].start()]
            )
            if attributes is None or not re.search(
                r"#\[(?:tokio::)?test(?:\(|\])", attributes.group()
            ):
                raise ValueError(
                    f"{identity}: {leaf!r} is not an attributed test; link an executable acceptance test"
                )
        elif case["kind"] != "browser":
            raise ValueError(f"{identity}: unsupported evidence kind {case['kind']!r}")
        for marker in case.get("source_markers", []):
            if marker not in source:
                raise ValueError(
                    f"{identity}: assertion marker {marker!r} disappeared; review the claimed behavior"
                )
    seen = set()
    for row in inventory["routes"]:
        key = (row["method"], row["path"])
        if key in seen:
            raise ValueError(f"duplicate behavioral contract {key}")
        seen.add(key)
        if row["scope"] != (
            "external_harness" if key in EXTERNAL_HARNESS else "server"
        ):
            raise ValueError(
                f"{key}: scope contradicts the explicit external-harness boundary"
            )
        if not row.get("assertions") or not row.get("remaining"):
            raise ValueError(
                f"{key}: state covered assertions and explicit remaining acceptance; no blanket parity claims"
            )
        for identity in row["evidence"]:
            if identity not in evidence:
                raise ValueError(f"{key}: missing evidence {identity}")
        if row["scope"] == "server" and not row["evidence"]:
            raise ValueError(
                f"{key}: no behavioral evidence linked; add acceptance or explicitly record a missing test"
            )
    expected = {(row["method"], row["path"]) for row in rows}
    if seen != expected:
        raise ValueError(
            f"behavioral route inventory differs: missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    categories = inventory["acceptance"]
    if {item["id"] for item in categories} != ACCEPTANCE or len(categories) != len(
        ACCEPTANCE
    ):
        raise ValueError(
            "original acceptance categories are missing/duplicated; preserve requirements beyond routes"
        )
    for item in categories:
        if not item.get("requirement") or not item.get("remaining"):
            raise ValueError(
                f"{item['id']}: preserve the full requirement and concrete unfinished work"
            )
        for identity in item["evidence"]:
            if identity not in evidence:
                raise ValueError(f"{item['id']}: unknown evidence {identity}")
    return inventory


def link(case: dict) -> str:
    return f"[{case.get('test', case['source'])}]({case['source'] if case['source'].startswith('../') else '../' + case['source']})"


def render(inventory: dict) -> str:
    lines = [
        "# llama.cpp behavioral acceptance matrix",
        "",
        "Generated by `python3 scripts/audit_llama_server_conformance.py --write`.",
        "`--check` validates inventory/source links; it executes no inference and proves no parity.",
        f"Upstream reference: `{inventory['upstream_commit']}`.",
        "",
        "This matrix records concrete test assertions and unfinished requirements. Source-linked",
        "tests are **not fresh passing results**. Optional `--run` produces an external receipt",
        "with exact test identities, positive pass counts, revision/content hashes and build settings.",
        "A passing fixture proves its listed assertions, not general backend/model or upstream parity.",
        "",
        "## Original acceptance requirements",
        "",
        "| Requirement | Linked cases | Work still required |",
        "| --- | --- | --- |",
    ]
    for item in inventory["acceptance"]:
        lines.append(
            f"| {item['requirement']} | {', '.join('`' + x + '`' for x in item['evidence'])} | {'; '.join(item['remaining'])} |"
        )
    lines += [
        "",
        "## Pinned endpoint contracts",
        "",
        "All 55 distinct method/path contracts are accounted for, including the four external-harness exclusions.",
        "The [registration audit](LLAMA_CPP_ENDPOINT_MATRIX.md) separately checks handler presence.",
        "",
        "| Contract | Scope | Linked assertion coverage | Evidence IDs | Remaining acceptance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in inventory["routes"]:
        lines.append(
            f"| `{row['method']} {row['path']}` | {row['scope']} | {row['assertions']} | {', '.join('`' + x + '`' for x in row['evidence'])} | {'; '.join(row['remaining'])} |"
        )
    lines += [
        "",
        "## Executable evidence definitions",
        "",
        "Rust cases name an explicit local package/manifest and one exact lib/integration test. Browser evidence is linked to its actual",
        "driver and must be run separately; the Rust runner rejects it instead of inventing a pass.",
        "",
        "| ID | Package | Source/test | Execution contract | Assertions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for identity, case in sorted(inventory["evidence"].items()):
        lines.append(
            f"| `{identity}` | {case.get('package', 'separate browser driver')} | {link(case)} | {case['execution_contract']} | {case['assertions']} |"
        )
    lines += [
        "",
        "## Running and interpreting evidence",
        "",
        "```sh",
        "python3 scripts/audit_llama_server_conformance.py --check",
        "# Run under the host's shared workload lock; builds may use substantial disk/RAM.",
        "python3 scripts/audit_llama_server_conformance.py --run aliases_responses aliases_stream aliases_counts \\",
        "  --offline --cargo-config /path/to/local-cargo-config.toml --receipt /tmp/llama-conformance.json",
        "```",
        "",
        "The runner requires a clean worktree, resolves `--list --exact` to exactly one test,",
        "then requires exactly one passed / zero failed / zero ignored test. Renamed, filtered-out,",
        "ignored, failed or ambiguous tests cannot produce a passing receipt. A partial/failed run",
        "is recorded as such. Receipts/logs must be outside the repository and include the inventory",
        "hash, HEAD/tree, source hashes, each selected package manifest, feature configuration,",
        "Rust/Cargo versions and exact per-package commands. Static checks verify manifest name,",
        "workspace membership, source containment and lib/integration target layout; exact runtime",
        "discovery still proves the compiled test identity. Build flags apply to each case's package.",
        "Cargo config discovery records the selected CARGO_HOME (or HOME), each cwd ancestor,",
        "both config/config.toml candidates (including missing sentinels), legacy-name precedence,",
        "explicit config, recursive include paths and their content hashes/path contexts. The",
        "inherited environment has an aggregate digest and selected build-input digests; values",
        "and config contents are not disclosed. Config/env changes during a run invalidate it.",
        "This snapshots inputs and discovery precedence; it does not reproduce Cargo's merged",
        "configuration or hash arbitrary compiler/wrapper executables and external dependencies.",
        "See [Cargo configuration](https://doc.rust-lang.org/cargo/reference/config.html).",
        "The checker requires Python 3.11+; runtime collection also requires POSIX. Each Cargo invocation owns",
        "a new process session/group. Timeout, SIGINT/SIGTERM/SIGHUP or another exception",
        "terminates that group (TERM then KILL), waits for Cargo and records a failed receipt.",
        "This contains local descendants retaining the group, not deliberately detached/remote",
        "runners; callers still own the shared workload lock. SIGKILL/system crashes are outside",
        "Python cleanup guarantees.",
        "The execution contract column describes the test code; the receipt does not independently",
        "instrument arbitrary devices or prove an untested backend. Keep device/performance claims separate.",
        "",
    ]
    return "\n".join(lines)


def target_args(case: dict) -> list[str]:
    return (
        ["--lib"]
        if case["target"] == "lib"
        else ["--test", case["target"].split(":", 1)[1]]
    )


def listed_test(output: str, expected: str) -> None:
    found = re.findall(r"^([^\n]+): test$", output, flags=re.MULTILINE)
    if found != [expected]:
        raise ValueError(
            f"test discovery selected {found!r}, expected exactly {expected!r}; repair the selector/features before claiming coverage"
        )


def passed_test(output: str, expected: str) -> None:
    summaries = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", output
    )
    if summaries != [("1", "0", "0")] or not re.search(
        r"^test " + re.escape(expected) + r" \.\.\. ok$", output, re.MULTILINE
    ):
        raise ValueError(
            f"{expected}: require exactly one passed, zero failed/ignored test; inspect the saved cargo output"
        )


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_state(root: Path) -> dict:
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "runtime evidence requires a clean worktree; commit/review changes and put receipts outside the repository"
        )
    return {
        "revision": git(root, "rev-parse", "HEAD"),
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
    }


# This is a configuration/input snapshot, not an implementation of Cargo's
# merge engine or a claim to hash arbitrary executable/compiler contents.
def cargo_context(
    root: Path, explicit: Path | None, environment: dict[str, str]
) -> dict:
    cwd = root.resolve()
    configured_home = environment.get("CARGO_HOME")
    if configured_home:
        home = Path(configured_home)
        home = home if home.is_absolute() else cwd / home
        home_origin = "CARGO_HOME"
    else:
        user_home = environment.get("HOME") or environment.get("USERPROFILE")
        home = (Path(user_home) if user_home else Path.home()) / ".cargo"
        home_origin = "HOME/USERPROFILE" if user_home else "OS home directory"
    groups = []
    for kind, directory in [
        ("cargo_home", home),
        *[("ancestor", parent / ".cargo") for parent in reversed([cwd, *cwd.parents])],
    ]:
        candidates = []
        for name in ("config", "config.toml"):
            path = directory / name
            candidates.append(
                {
                    "path": str(path.absolute()),
                    "resolved_path": str(path.resolve()),
                    "sha256": digest(path) if path.is_file() else None,
                    "exists": path.exists(),
                }
            )
        selected = next((item["path"] for item in candidates if item["exists"]), None)
        groups.append(
            {
                "origin": kind,
                "path_context": str(directory.parent.absolute()),
                "candidates": candidates,
                "selected": selected,
            }
        )
    if explicit is not None:
        # Match the exact resolved path passed to Cargo's --config argument;
        # includes and relative config values belong to that file's context.
        path = explicit.resolve()
        groups.append(
            {
                "origin": "explicit_cli",
                "supplied_path": str(explicit.absolute()),
                "path_context": str(path.parent.parent),
                "candidates": [
                    {
                        "path": str(path),
                        "resolved_path": str(path.resolve()),
                        "sha256": digest(path) if path.is_file() else None,
                        "exists": path.exists(),
                    }
                ],
                "selected": str(path),
            }
        )
    included = []

    def includes(path: Path, ancestry: tuple[Path, ...]) -> None:
        resolved = path.resolve()
        if resolved in ancestry:
            raise ValueError(
                f"Cargo configuration include cycle at {path}; repair it before collecting evidence"
            )
        if not path.is_file():
            return
        document = tomllib.loads(path.read_text())
        imports = document.get("include", [])
        if not isinstance(imports, list):
            raise TypeError(
                f"Cargo configuration {path} include must be an array of paths/tables"
            )
        for item in imports:
            name = (
                item
                if isinstance(item, str)
                else item.get("path")
                if isinstance(item, dict)
                else None
            )
            if not isinstance(name, str):
                raise TypeError(
                    f"Cargo configuration {path} has an unsupported include; supply a path string/table"
                )
            child = path.parent / name
            included.append(
                {
                    "included_by": str(path),
                    "path": str(child.absolute()),
                    "resolved_path": str(child.resolve()),
                    "path_context": str(child.parent.parent.absolute()),
                    "sha256": digest(child) if child.is_file() else None,
                    "exists": child.exists(),
                    "optional": item.get("optional", False)
                    if isinstance(item, dict)
                    else False,
                }
            )
            includes(child, (*ancestry, resolved))

    for group in groups:
        if group["selected"] is not None:
            includes(Path(group["selected"]), ())
    names = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "CARGO_HOME",
        "RUSTFLAGS",
        "RUSTDOCFLAGS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CC",
        "CXX",
        "AR",
        "CFLAGS",
        "CPPFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    }
    names.update(
        name
        for name in environment
        if name.startswith(
            (
                "CARGO_",
                "RUST",
                "ONNX_GENAI_",
                "ORT_",
                "CC_",
                "CXX_",
                "AR_",
                "CMAKE_",
                "PKG_CONFIG",
                "CUDA_",
                "CUDNN_",
                "CUBLAS_",
                "NVCC",
            )
        )
    )
    return {
        "cwd": str(cwd),
        "cargo_home": str(home.absolute()),
        "cargo_home_origin": home_origin,
        "precedence": "groups low to high, then environment, then explicit CLI config; config wins over config.toml in one directory; include paths left to right before their parent",
        "groups_low_to_high": groups,
        "included_configs": included,
        "environment_sha256": hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest(),
        "build_environment_sha256": {
            name: hashlib.sha256(environment[name].encode()).hexdigest()
            if name in environment
            else None
            for name in sorted(names)
        },
    }


class RunInterrupted(KeyboardInterrupt):
    """A catchable interruption that still permits containment cleanup/receipt."""


def run_contained(
    command: list[str], *, cwd: Path, stdout, timeout: int, env: dict[str, str]
) -> subprocess.CompletedProcess:
    if os.name != "posix":
        raise ValueError(
            "runtime receipts require a POSIX host with process-group containment; --check remains available on other hosts"
        )
    watched = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    previous = {number: signal.getsignal(number) for number in watched}

    process = None
    pending_signal = None

    def interrupted(number, _frame):
        nonlocal pending_signal
        # Finish obtaining the owned Popen handle if cancellation arrives during
        # fork/exec; unwinding before assignment could lose the child handle.
        if process is None:
            pending_signal = number
            return
        raise RunInterrupted(f"interrupted by signal {number}")

    try:
        for number in watched:
            signal.signal(number, interrupted)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        if pending_signal is not None:
            raise RunInterrupted(
                f"interrupted by signal {pending_signal} during process creation"
            )
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    finally:
        # Cleanup must finish before returning control to the external lock owner.
        # Ignore repeated cancellation while terminating this owned group.
        try:
            for number in watched:
                signal.signal(number, signal.SIG_IGN)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def run(inventory: dict, args: argparse.Namespace, root: Path = ROOT) -> None:
    receipt = args.receipt.resolve()
    if receipt.is_relative_to(root.resolve()):
        raise ValueError(
            "receipt/logs must be outside the repository; select e.g. /tmp/llama-conformance.json"
        )
    if not args.run:
        raise ValueError(
            "select at least one Rust evidence case; an empty run cannot prove coverage"
        )
    cases = []
    for identity in dict.fromkeys(args.run):
        case = inventory["evidence"].get(identity)
        if case is None or case["kind"] != "rust":
            raise ValueError(
                f"{identity}: select a defined Rust case; browser tests require the documented browser driver"
            )
        cases.append((identity, case))
    state = source_state(root)
    sources = {
        str(INVENTORY),
        "scripts/audit_llama_server_conformance.py",
        "Cargo.lock",
        "Cargo.toml",
    }
    sources.update(case["source"] for _, case in cases)
    sources.update(case["package_manifest"] for _, case in cases)
    hashes = {name: digest(root / name) for name in sorted(sources)}
    command = ["cargo", "test", "-j", str(args.jobs)]
    if args.offline:
        command.append("--offline")
    if args.no_default_features:
        command.append("--no-default-features")
    if args.features:
        command += ["--features", args.features]
    environment = dict(os.environ)
    context = cargo_context(root, args.cargo_config, environment)
    if args.cargo_config:
        command += ["--config", context["groups_low_to_high"][-1]["selected"]]
    result = {
        "version": 1,
        "status": "running",
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": state,
        "content_sha256": hashes,
        "features": {
            "defaults": not args.no_default_features,
            "additional": args.features,
        },
        "cargo_context": context,
        "tools": {},
        "tool_invocations": {},
        "cases": [],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = receipt.with_suffix(receipt.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(receipt)

    save()
    try:
        for tool in ("cargo", "rustc"):
            invocation = [tool, "--version"]
            log = receipt.with_name(receipt.stem + f".{tool}.version.log")
            result["tool_invocations"][tool] = {"command": invocation, "log": str(log)}
            save()
            with log.open("w") as output:
                completed = run_contained(
                    invocation,
                    cwd=root,
                    stdout=output,
                    timeout=min(30, args.timeout_seconds),
                    env=environment,
                )
            if completed.returncode:
                raise ValueError(
                    f"{tool} version query exited {completed.returncode}; inspect {log}"
                )
            result["tools"][tool] = log.read_text().strip()
            save()
        for identity, case in cases:
            entry = {
                "id": identity,
                "test": case["test"],
                "package": case["package"],
                "package_manifest": case["package_manifest"],
                "target": case["target"],
                "execution_contract": case["execution_contract"],
                "status": "running",
            }
            result["cases"].append(entry)
            save()
            for phase, tail, check in [
                ("list", ["--exact", "--list"], listed_test),
                ("run", ["--exact", "--test-threads=1"], passed_test),
            ]:
                invocation = [
                    *command,
                    "-p",
                    case["package"],
                    *target_args(case),
                    case["test"],
                    "--",
                    *tail,
                ]
                log = receipt.with_name(receipt.stem + f".{identity}.{phase}.log")
                entry[phase] = {"command": invocation, "log": str(log)}
                save()
                with log.open("w") as output:
                    completed = run_contained(
                        invocation,
                        cwd=root,
                        stdout=output,
                        timeout=args.timeout_seconds,
                        env=environment,
                    )
                if completed.returncode:
                    raise ValueError(
                        f"{identity} {phase} exited {completed.returncode}; inspect {log}"
                    )
                check(log.read_text(), case["test"])
                config_changed = (
                    cargo_context(root, args.cargo_config, dict(os.environ)) != context
                )
                if (
                    config_changed
                    or source_state(root) != state
                    or any(
                        digest(root / name) != expected
                        for name, expected in hashes.items()
                    )
                ):
                    raise ValueError(
                        "source/config changed during runtime evidence collection; re-review and rerun"
                    )
            entry.update(status="passed", passed=1, failed=0, ignored=0)
            save()
            print(f"{identity}: 1 passed; exact test {case['test']}", flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if result["cases"]:
            result["cases"][-1]["status"] = "failed"
        raise
    finally:
        result["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--run", nargs="+", metavar="EVIDENCE_ID")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--cargo-config", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument("--features", default="")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    try:
        inventory = validate(ROOT, json.loads((ROOT / INVENTORY).read_text()))
        if args.run:
            if args.receipt is None or args.jobs <= 0 or args.timeout_seconds <= 0:
                parser.error(
                    "--run requires --receipt outside the repository and positive jobs/timeout"
                )
            run(inventory, args)
        else:
            report = render(inventory)
            if args.write:
                (ROOT / REPORT).write_text(report)
            elif not (ROOT / REPORT).exists() or (ROOT / REPORT).read_text() != report:
                raise ValueError(
                    "behavioral matrix is stale; run --write, review its evidence/gaps and commit it"
                )
            print(
                f"source inventory checked: {len(inventory['routes'])} contracts, {len(inventory['acceptance'])} acceptance categories; runtime execution not performed"
            )
    except KeyboardInterrupt:
        parser.exit(
            130,
            "conformance command interrupted; any managed Cargo group was stopped; inspect the runtime receipt if collection started\n",
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
