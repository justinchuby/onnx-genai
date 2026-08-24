#!/usr/bin/env python3
"""Verify CUDA integration tests cannot silently pass on CPU-only runners."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_CRATE = ROOT / "crates" / "onnx-runtime-ep-cuda"
# Device memory moved to its own crate so that using an allocator does not drag
# in every kernel. Its GPU tests moved with it, and a check naming only the
# execution provider would stop seeing them -- which is precisely the hole this
# script exists to close.
MEMORY_CRATE = ROOT / "crates" / "onnx-runtime-cuda-memory"
CUDA_CRATES = (CUDA_CRATE, MEMORY_CRATE)
TESTS = CUDA_CRATE / "tests"
TEST_DIRS = tuple(crate / "tests" for crate in CUDA_CRATES)
# CUDA integration-test targets conventionally end in `_gpu`. Keep historical
# CUDA targets that predate that convention explicit so a genuinely CPU-only
# target is not policed as a device test merely because it lives in a CUDA
# crate.
CUDA_TARGETS_WITHOUT_SUFFIX = frozenset({"matmul_nbits_marlin_numerics"})

# CUDA-named targets that must run in every configuration, and so cannot be
# held to the "ignored, not passed" rule the rest of the suite is checked
# against.
#
# `suite_canary_gpu` is the test that exists because the rest of the suite can
# skip silently. It is a no-op unless `NXRT_REQUIRE_CUDA` says a GPU is meant to
# be present, and where that is set it fails loudly. Giving it the `gpu-tests`
# ignore would remove it from exactly the runs it was written to police -- a
# CPU-only machine that believes it tested a GPU.
#
ALWAYS_RUN = frozenset({"suite_canary_gpu"})
SUMMARY = re.compile(
    r"test result: (?:ok|FAILED)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; (?P<filtered>\d+) filtered out"
)

# Per-test outcome lines. A count alone ("1 tests failed while checking ignored
# status") does not say which test, and this checker's whole job is to be the
# signal that a `_gpu` test was added without an ignore -- so it has to name the
# test, or the next person pays for a bisect to find out what it meant.
OUTCOME = re.compile(r"^test (?P<name>\S+) \.\.\. (?P<status>ok|FAILED|ignored)", re.MULTILINE)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    features: str


@dataclass(frozen=True)
class TestBinary:
    target: str
    executable: Path


@dataclass(frozen=True)
class LibtestResult:
    target: str
    inventory: int
    passed: int
    failed: int
    ignored: int
    names: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def named(self, status: str) -> str:
        """`" (a, b)"` for the tests with `status`, or `""` if unknown.

        Appended to a count so the message says which test, while still being
        correct if the per-test lines could not be parsed.
        """
        for key, values in self.names:
            if key == status and values:
                return " (" + ", ".join(sorted(values)) + ")"
        return ""


@dataclass(frozen=True)
class IgnoredResult(LibtestResult):
    pass


@dataclass(frozen=True)
class ActiveResult(LibtestResult):
    pass


BASE_CONFIG = FeatureConfig("without-gpu-tests", "cuda")
GPU_CONFIG = FeatureConfig("with-gpu-tests", "cuda,gpu-tests")


def run(command: list[str | Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def is_cuda_test_target(target: str) -> bool:
    return (
        target not in ALWAYS_RUN
        and (target.endswith("_gpu") or target in CUDA_TARGETS_WITHOUT_SUFFIX)
    )


# `ignore` as an attribute token, not as English. The first version of this
# scan searched the item's head for the substring "ignore", and so passed on a
# test whose doc comment happened to contain the words "a kernel that ignored
# the attribute" -- the exact defect it was written to catch. Anchor on the
# punctuation an attribute must have around it, and never look at doc text.
IGNORE_ATTR = re.compile(r"(?:^|[\[(,\s])ignore\s*(?:=|\]|,|\)|$)")


def attribute_lines_of_item(lines: list[str], index: int) -> list[str]:
    """The attribute lines belonging to the `#[test]` item at `lines[index]`.

    Collects the contiguous attribute block above the marker and any attributes
    between it and the `fn`, since attribute order is not significant in Rust.
    Doc comments are excluded: they are prose, and reading them as though they
    were attributes is what made the first version of this check vacuous.
    """
    attributes: list[str] = []
    pending = 0
    for line in reversed(lines[:index]):
        stripped = line.strip()
        if not stripped:
            break
        if pending == 0 and stripped.startswith("//"):
            continue
        delta = stripped.count("]") + stripped.count(")") - stripped.count("[") - stripped.count("(")
        # A multi-line attribute is met tail-first walking upward, so a line
        # that closes more brackets than it opens starts a continuation.
        if pending == 0 and delta <= 0 and not stripped.startswith("#["):
            break
        attributes.append(stripped)
        pending = max(0, pending + delta)
    pending = 0
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if pending == 0:
            if stripped.startswith("//"):
                continue
            if not stripped.startswith("#["):
                break
        attributes.append(stripped)
        pending = max(0, pending + stripped.count("[") + stripped.count("(") - stripped.count("]") - stripped.count(")"))
    return attributes


def test_name_at(lines: list[str], index: int) -> str:
    for line in lines[index + 1 : index + 12]:
        match = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)", line.strip())
        if match:
            return match.group(1)
    return "<unknown fn>"


def un_ignored_tests(source: str) -> list[tuple[int, str]]:
    """Every `#[test]` in `source` that carries no ignore attribute, 1-based."""
    lines = source.splitlines()
    found: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if line.strip() != "#[test]":
            continue
        attributes = attribute_lines_of_item(lines, index)
        if any(IGNORE_ATTR.search(entry) for entry in attributes):
            continue
        found.append((index + 1, test_name_at(lines, index)))
    return found


def scan_source_for_missing_ignores() -> list[str]:
    """Every `#[test]` in a policed CUDA target must carry an ignore.

    The inventory check in this script is authoritative, but it needs two full
    CUDA test builds, so it runs only on the CUDA lane -- which means feedback
    on the single most common way to break that lane arrives after merge, and
    only if the lane is green enough to be believed. Five tests were added
    without an ignore in three days, each one reported by a red lane that was
    already red for an unrelated reason.

    This is the same rule read straight off the source in well under a second
    with no CUDA toolchain, so it can run on every pull request. It does not
    replace the inventory check: it cannot see macro-expanded tests, and it
    cannot check inventory parity across the two feature configurations. It
    catches one specific recurring omission early.
    """
    errors: list[str] = []
    for test_dir in TEST_DIRS:
        for path in sorted(test_dir.glob("*.rs")):
            if not is_cuda_test_target(path.stem):
                continue
            source = path.read_text(encoding="utf-8")
            for line_number, name in un_ignored_tests(source):
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {name} "
                    "has no ignore attribute. Every test in a CUDA target must be ignored "
                    "when gpu-tests is off, or it runs on a machine with no device: add "
                    '#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]'
                )
    return errors


def parse_test_binaries_from_json(stdout: str) -> list[TestBinary]:
    binaries: dict[str, Path] = {}
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target", {})
        if "test" not in target.get("kind", []):
            continue
        src_path = Path(target.get("src_path", ""))
        if not src_path.is_absolute():
            src_path = ROOT / src_path
        if not any(src_path.is_relative_to(tests) for tests in TEST_DIRS):
            continue
        executable = message.get("executable")
        if executable and is_cuda_test_target(target["name"]):
            binaries[target["name"]] = Path(executable)

    return [
        TestBinary(target, binaries[target])
        for target in sorted(binaries)
        if target not in ALWAYS_RUN
    ]


def build_test_binaries(config: FeatureConfig) -> list[TestBinary]:
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            "onnx-runtime-ep-cuda",
            "-p",
            "onnx-runtime-cuda-memory",
            "--features",
            config.features,
            "--tests",
            "--no-run",
            "--message-format=json",
        ]
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"cargo test --no-run failed for {config.name}")
    return parse_test_binaries_from_json(result.stdout)


def list_inventory(binary: TestBinary) -> frozenset[str]:
    result = run([binary.executable, "--list"])
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} --list failed")
    return frozenset(line.removesuffix(": test") for line in result.stdout.splitlines() if line.endswith(": test"))


def run_libtest(binary: TestBinary) -> tuple[int, int, int, int, dict[str, list[str]]]:
    result = run([binary.executable, "--color", "never"])
    output = result.stdout + result.stderr
    matches = list(SUMMARY.finditer(output))
    if not matches:
        print(output, file=sys.stderr, end="")
        raise RuntimeError(f"{binary.target} did not print a libtest summary")
    summary = matches[-1]
    names: dict[str, list[str]] = {"ok": [], "FAILED": [], "ignored": []}
    for outcome in OUTCOME.finditer(output):
        names[outcome.group("status")].append(outcome.group("name"))
    return (
        result.returncode,
        int(summary.group("passed")),
        int(summary.group("failed")),
        int(summary.group("ignored")),
        names,
    )


def collect_inventories(binaries: list[TestBinary]) -> dict[str, frozenset[str]]:
    return {binary.target: list_inventory(binary) for binary in binaries}


def compare_inventories(
    base: dict[str, frozenset[str]], gpu: dict[str, frozenset[str]]
) -> list[str]:
    errors: list[str] = []
    base_targets = set(base)
    gpu_targets = set(gpu)
    for target in sorted(gpu_targets - base_targets):
        errors.append(f"{target}: exists only with gpu-tests enabled; CUDA tests must not hide from CPU inventory")
    for target in sorted(base_targets - gpu_targets):
        errors.append(f"{target}: exists only without gpu-tests enabled; inventories must stay reconciled")
    for target in sorted(base_targets & gpu_targets):
        base_only = base[target] - gpu[target]
        gpu_only = gpu[target] - base[target]
        for test in sorted(gpu_only):
            errors.append(f"{target}::{test}: test exists only with gpu-tests enabled")
        for test in sorted(base_only):
            errors.append(f"{target}::{test}: test exists only without gpu-tests enabled")
    return errors


def known_names(
    names: dict[str, list[str]], inventory: frozenset[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Keep only outcome names Cargo actually inventoried for the target.

    libtest reprints a failing test's captured stdout verbatim at column 0, so a
    test that itself printed a line shaped like `test evil ... ok` would be
    parsed as an outcome. That can only ever mislabel the annotation -- every
    error here is decided by counts, never by names -- but an error message that
    names a test which does not exist is worse than one that names none, so the
    parse is intersected with the inventory rather than trusted.
    """
    return tuple(
        (status, tuple(name for name in found if name in inventory))
        for status, found in names.items()
    )


def validate_ignored_result(result: IgnoredResult) -> list[str]:
    errors: list[str] = []
    if result.inventory == 0:
        errors.append(f"{result.target}: Cargo reported no integration tests")
    if result.failed:
        errors.append(
            f"{result.target}: {result.failed} tests failed while checking ignored status"
            f"{result.named('FAILED')}"
        )
    if result.passed:
        errors.append(
            f"{result.target}: {result.passed} tests executed without gpu-tests; "
            f"CUDA tests must be ignored, not pass{result.named('ok')}"
        )
    if result.ignored != result.inventory:
        errors.append(
            f"{result.target}: Cargo inventory has {result.inventory} tests but libtest reported {result.ignored} ignored"
        )
    return errors


def validate_active_no_cuda_result(result: ActiveResult) -> list[str]:
    errors: list[str] = []
    if result.inventory == 0:
        errors.append(f"{result.target}: Cargo reported no gpu-tests integration tests")
    if result.passed:
        errors.append(
            f"{result.target}: {result.passed} tests passed with gpu-tests on a no-CUDA host; "
            f"CUDA tests must fail loud or remain ignored{result.named('ok')}"
        )
    if result.failed + result.ignored != result.inventory:
        errors.append(
            f"{result.target}: Cargo inventory has {result.inventory} tests but active no-CUDA run reported "
            f"{result.failed} failed + {result.ignored} ignored"
        )
    return errors


def declares_gpu_tests_feature(manifest: str) -> bool:
    """True if the manifest defines a `gpu-tests` feature, whatever its value.

    The property this guard cares about is that the feature *exists*, so that
    `--features gpu-tests` is a meaningful flag for the crate. An earlier
    version matched the literal string `gpu-tests = []`, which silently
    conflated "declares the feature" with "declares it with an empty value" and
    so rejected the legitimate forwarding form
    `gpu-tests = ["onnx-runtime-cuda-memory/gpu-tests"]`.

    The key is looked for only inside the `[features]` table. A file-wide match
    would accept a *dependency* named `gpu-tests`, or one under a
    `[target.'cfg(...)'.dependencies]` table -- which is the same mistake in a
    new costume: matching a string that usually co-occurs with the property
    instead of the property itself.
    """
    in_features = False
    for raw_line in manifest.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_features = line.split("#", 1)[0].strip() == "[features]"
            continue
        if in_features and re.match(r"gpu-tests\s*=", line):
            return True
    return False


def self_test() -> None:
    for manifest, expected in (
        ("[features]\ngpu-tests = []\n", True),
        ('[features]\ngpu-tests = ["onnx-runtime-cuda-memory/gpu-tests"]\n', True),
        ("[features]\ngpu-tests = [\n    \"dep/gpu-tests\",\n]\n", True),
        ('[dependencies]\nfoo = "1"\n\n[features]\ngpu-tests = []\n', True),
        ('[features]\ngpu-tests = []\n\n[dependencies]\nfoo = "1"\n', True),
        ("[features] # the table\ngpu-tests = []\n", True),
        ("[features]\ndefault = []\n", False),
        ("[features]\n# gpu-tests = []\n", False),
        ("[features]\ngpu-tests-extra = []\n", False),
        ("[dependencies]\nfoo = { features = [\"gpu-tests\"] }\n", False),
        ('[dependencies]\ngpu-tests = "1.0"\n', False),
        ('[target.\'cfg(unix)\'.dependencies]\ngpu-tests = "1.0"\n', False),
    ):
        if declares_gpu_tests_feature(manifest) is not expected:
            raise AssertionError(f"gpu-tests feature detection wrong for: {manifest!r}")

    named = IgnoredResult(
        "activations_gpu",
        inventory=4,
        passed=0,
        failed=1,
        ignored=3,
        names=(("FAILED", ("mish_matches_cpu_including_the_saturating_tail",)),),
    )
    messages = validate_ignored_result(named)
    if not any("mish_matches_cpu_including_the_saturating_tail" in m for m in messages):
        raise AssertionError(f"failing test was not named in {messages!r}")
    if not any("Cargo inventory has 4 tests but libtest reported 3 ignored" in m for m in messages):
        raise AssertionError(f"ignored/inventory mismatch not reported in {messages!r}")

    # Unparsed per-test lines must degrade to the count, not to a crash or a
    # message with an empty "()" in it.
    unnamed = IgnoredResult("t_gpu", inventory=1, passed=1, failed=0, ignored=0)
    if any("(" in m for m in validate_ignored_result(unnamed)):
        raise AssertionError("empty name list must not render parentheses")

    active = ActiveResult(
        "t_gpu",
        inventory=2,
        passed=1,
        failed=0,
        ignored=1,
        names=(("ok", ("sneaky_pass",)),),
    )
    if not any("sneaky_pass" in m for m in validate_active_no_cuda_result(active)):
        raise AssertionError("passing test was not named in the active-config message")

    phantom = known_names(
        {"FAILED": ["real_test", "evil"], "ok": [], "ignored": []},
        frozenset({"real_test", "other_test"}),
    )
    if dict(phantom)["FAILED"] != ("real_test",):
        raise AssertionError(f"uninventoried name was not dropped: {phantom!r}")

    parsed = {
        outcome.group("status"): outcome.group("name")
        for outcome in OUTCOME.finditer(
            "running 3 tests\n"
            "test alpha ... ok\n"
            "test beta ... FAILED\n"
            "test gamma ... ignored\n"
        )
    }
    if parsed != {"ok": "alpha", "FAILED": "beta", "ignored": "gamma"}:
        raise AssertionError(f"libtest outcome parsing wrong: {parsed!r}")

    for cpu_target in (
        "deferred_release_queue",
        "no_built_in_eager_allocator",
        "vmm_release_quarantine",
    ):
        if is_cuda_test_target(cpu_target):
            raise AssertionError(f"CPU-only target {cpu_target} was classified as CUDA")
    if not is_cuda_test_target("matmul_nbits_marlin_numerics"):
        raise AssertionError("historical CUDA target without _gpu suffix was not classified as CUDA")

    fixture_stdout = "\n".join(
        [
            "not json",
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "fixture_gpu",
                        "kind": ["test"],
                        "src_path": str(TESTS / "fixture_gpu.rs"),
                    },
                    "executable": str(ROOT / "target" / "debug" / "deps" / "fixture_gpu.exe"),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "deferred_release_queue",
                        "kind": ["test"],
                        "src_path": str(TESTS / "deferred_release_queue.rs"),
                    },
                    "executable": str(
                        ROOT / "target" / "debug" / "deps" / "deferred_release_queue.exe"
                    ),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "matmul_nbits_marlin_numerics",
                        "kind": ["test"],
                        "src_path": str(TESTS / "matmul_nbits_marlin_numerics.rs"),
                    },
                    "executable": str(
                        ROOT / "target" / "debug" / "deps" / "matmul_nbits_marlin_numerics.exe"
                    ),
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {
                        "name": "onnx_runtime_ep_cuda",
                        "kind": ["lib"],
                        "src_path": str(CUDA_CRATE / "src" / "lib.rs"),
                    },
                    "executable": str(ROOT / "target" / "debug" / "deps" / "lib.exe"),
                }
            ),
        ]
    )
    parsed = parse_test_binaries_from_json(fixture_stdout)
    expected = [
        TestBinary("fixture_gpu", ROOT / "target" / "debug" / "deps" / "fixture_gpu.exe"),
        TestBinary(
            "matmul_nbits_marlin_numerics",
            ROOT / "target" / "debug" / "deps" / "matmul_nbits_marlin_numerics.exe",
        ),
    ]
    if parsed != expected:
        raise AssertionError(f"JSON parser fixture returned {parsed!r}")

    if compare_inventories({"target": frozenset({"a"})}, {"target": frozenset({"a"})}):
        raise AssertionError("matching inventories should pass")
    hidden = compare_inventories({"target": frozenset({"a"})}, {"target": frozenset({"a", "gpu_only"})})
    if not any("gpu_only" in error for error in hidden):
        raise AssertionError("gpu-tests-only inventory drift should be caught")

    good_ignored = IgnoredResult("fixture_good", inventory=2, passed=0, failed=0, ignored=2)
    silent_without_feature = IgnoredResult("fixture_silent_skip", inventory=1, passed=1, failed=0, ignored=0)
    if validate_ignored_result(good_ignored):
        raise AssertionError("good ignored fixture should pass")
    if not any("executed without gpu-tests" in error for error in validate_ignored_result(silent_without_feature)):
        raise AssertionError("without-gpu-tests silent pass should fail")

    # The first version of this scan searched the item head for the substring
    # "ignore" and so accepted a test whose doc comment said "a kernel that
    # ignored the attribute". That is the defect it exists to catch, and it is
    # the second fixture below. Both controls are here on purpose: the ones
    # that must be flagged, and the ones that must not.
    for source, expected in (
        ("#[test]\nfn a() {}\n", ["a"]),
        ("/// a kernel that ignored the attribute entirely\n#[test]\nfn a() {}\n", ["a"]),
        ("/// close for small |x|). We run\n#[test]\nfn a() {}\n", ["a"]),
        ("#[cfg(feature = \"gpu-tests\")]\n#[test]\nfn a() {}\n", ["a"]),
        ("#[ignore]\n#[test]\nfn a() {}\n", []),
        ("#[test]\n#[ignore = \"needs a device\"]\nfn a() {}\n", []),
        ("#[cfg_attr(not(feature = \"gpu-tests\"), ignore = \"x\")]\n#[test]\nfn a() {}\n", []),
        ('''#[cfg_attr(\n    not(feature = "gpu-tests"),\n    ignore = "requires CUDA device"\n)]\n#[test]\nfn a() {}\n''', []),
        ("/// doc\n#[cfg_attr(not(feature = \"gpu-tests\"), ignore = \"x\")]\n#[test]\nfn a() {}\n", []),
        ("#[test]\nfn a() {}\n\n#[ignore]\n#[test]\nfn b() {}\n", ["a"]),
        ("macro_rules! m {\n    () => {\n        #[ignore]\n        #[test]\n        fn a() {}\n    };\n}\n", []),
        ("// let ignore = 1;\n#[test]\nfn a() {}\n", ["a"]),
        # These two exist to make the two defences falsifiable rather than
        # merely present. The first fails if IGNORE_ATTR is relaxed to a bare
        # substring: "ignore" appears, but inside a string literal, not as an
        # attribute. The second fails if doc comments are read as attributes:
        # the prose ends "ignore," which is attribute-shaped punctuation.
        ("#[should_panic(expected = \"ignore me\")]\n#[test]\nfn a() {}\n", ["a"]),
        ("/// we do not ignore, ever\n#[test]\nfn a() {}\n", ["a"]),
    ):
        names = [name for _, name in un_ignored_tests(source)]
        if names != expected:
            raise AssertionError(f"source scan fixture returned {names!r}, expected {expected!r}: {source!r}")

    good_active = ActiveResult("fixture_active", inventory=2, passed=0, failed=2, ignored=0)
    silent_with_feature = ActiveResult("fixture_active_silent", inventory=1, passed=1, failed=0, ignored=0)
    if validate_active_no_cuda_result(good_active):
        raise AssertionError("active fail-loud fixture should pass")
    if not any("passed with gpu-tests" in error for error in validate_active_no_cuda_result(silent_with_feature)):
        raise AssertionError("gpu-tests-enabled silent pass should fail")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run parser fixtures only")
    parser.add_argument(
        "--source-scan",
        action="store_true",
        help="check the source for un-ignored CUDA tests; no cargo build, runs anywhere",
    )
    args = parser.parse_args()

    self_test()
    if args.self_test:
        print("CUDA honesty checker self-test passed")
        return 0

    if args.source_scan:
        source_errors = scan_source_for_missing_ignores()
        if source_errors:
            print("CUDA test source scan failed:", file=sys.stderr)
            for error in source_errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("CUDA test source scan passed: every test in a CUDA target carries an ignore")
        return 0

    errors: list[str] = scan_source_for_missing_ignores()
    for crate in CUDA_CRATES:
        manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
        if not declares_gpu_tests_feature(manifest):
            errors.append(f"crates/{crate.name}/Cargo.toml must define a gpu-tests feature")

    base_binaries = build_test_binaries(BASE_CONFIG)
    gpu_binaries = build_test_binaries(GPU_CONFIG)
    base_inventory = collect_inventories(base_binaries)
    gpu_inventory = collect_inventories(gpu_binaries)
    errors.extend(compare_inventories(base_inventory, gpu_inventory))

    base_by_target = {binary.target: binary for binary in base_binaries}
    gpu_by_target = {binary.target: binary for binary in gpu_binaries}

    ignored_results: list[IgnoredResult] = []
    for target, inventory in sorted(base_inventory.items()):
        _, passed, failed, ignored, names = run_libtest(base_by_target[target])
        result = IgnoredResult(
            target,
            len(inventory),
            passed,
            failed,
            ignored,
            known_names(names, inventory),
        )
        ignored_results.append(result)
        errors.extend(validate_ignored_result(result))

    active_results: list[ActiveResult] = []
    for target, inventory in sorted(gpu_inventory.items()):
        _, passed, failed, ignored, names = run_libtest(gpu_by_target[target])
        result = ActiveResult(
            target,
            len(inventory),
            passed,
            failed,
            ignored,
            known_names(names, inventory),
        )
        active_results.append(result)
        errors.extend(validate_active_no_cuda_result(result))

    total_base_inventory = sum(len(inventory) for inventory in base_inventory.values())
    total_gpu_inventory = sum(len(inventory) for inventory in gpu_inventory.values())
    total_ignored = sum(result.ignored for result in ignored_results)
    total_active_failed = sum(result.failed for result in active_results)
    total_active_ignored = sum(result.ignored for result in active_results)
    if total_base_inventory == 0 or total_gpu_inventory == 0:
        errors.append("Cargo reported no CUDA integration tests")

    if errors:
        print("CUDA test honesty check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "CUDA test honesty check passed: "
        f"{total_base_inventory} tests/{len(base_inventory)} targets without gpu-tests "
        f"({total_ignored} ignored), {total_gpu_inventory} tests/{len(gpu_inventory)} targets with gpu-tests "
        f"({total_active_failed} fail-loud, {total_active_ignored} ignored, 0 passed on this no-CUDA host)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
