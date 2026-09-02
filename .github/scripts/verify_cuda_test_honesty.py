#!/usr/bin/env python3
"""Verify CUDA integration tests cannot silently pass on CPU-only runners."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
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
# Rust integration targets are separate processes, so no in-process mutex can
# serialize one target against another. These locks intentionally serialize
# only the libtest threads inside the named binary: that is where concurrent EP
# construction exhausts VMM address reservations and process-global telemetry
# deltas race. Keeping the lock local to each target makes that scope explicit.
#
# This set is the policy census, deliberately independent from the implementation
# mapping below. MHA, DFT, and STFT were added after parallel execution reproduced
# VMM/telemetry races; NMS already carried the same target-local isolation and is
# therefore part of the policy too. A mapping cannot remove its own obligation:
# the scanner first requires its keys to equal this non-empty set.
EXPECTED_SUITE_ISOLATION_TARGETS = frozenset(
    {
        "multi_head_attention_gpu",
        "dft_gpu",
        "stft_gpu",
        "nms_gpu",
    }
)
SERIALIZED_TARGET_LOCKS = {
    "multi_head_attention_gpu": ("lock_mha_gpu", "MHA_GPU_LOCK"),
    "dft_gpu": ("lock_dft_gpu", "DFT_GPU_LOCK"),
    "stft_gpu": ("lock_stft_gpu", "STFT_GPU_LOCK"),
    "nms_gpu": ("lock_nms_gpu", "NMS_GPU_LOCK"),
}
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
CUDA_PLUGIN_PACKAGE = "onnx-runtime-ep-cuda-plugin"
CUDA_PLUGIN_TARGET = "cuda_unique_ort_e2e"
CUDA_PLUGIN_FEATURES = "cuda"
CUDA_PLUGIN_MIN_TESTS = 3
SESSION_PACKAGE = "onnx-runtime-session"
SESSION_HETERO_TARGET = "hetero_cuda_gpu"
SESSION_BASE_FEATURES = "cuda,cuda-13000"
SESSION_GPU_FEATURES = "gpu-tests,cuda-13000"
SESSION_HETERO_MIN_TESTS = 1
ENGINE_PACKAGE = "onnx-genai-engine"
ENGINE_DEFAULT_OFF_TARGET = "native_cuda_default_off_state_capture_gpu"
ENGINE_BASE_FEATURES = "native-cuda,cuda-13000"
ENGINE_GPU_FEATURES = "gpu-tests,cuda-13000"
ENGINE_DEFAULT_OFF_MIN_TESTS = 1


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


@lru_cache(maxsize=1)
def policed_cuda_targets_by_dir() -> dict[Path, tuple[Path, ...]]:
    """The policed CUDA test targets, kept separated by the directory they came from.

    The census needs per-directory counts, not a total. These crates have
    split before -- the GPU tests moved out of the execution provider into
    `onnx-runtime-cuda-memory`, which is why TEST_DIRS names two directories
    -- and a total cannot distinguish "both present" from "one emptied and the
    other still large enough to look healthy".

    Cached, so the scans and the census read one enumeration rather than three
    of their own. They already shared the filter, which is what stops the
    census certifying a set the scans do not police; enumerating once makes
    that true of the run and not only of the logic. Tuples because a cached
    mutable result is a caller's mistake waiting to happen.
    """
    return {
        test_dir: tuple(
            path
            for path in sorted(test_dir.glob("*.rs"))
            if is_cuda_test_target(path.stem)
        )
        for test_dir in TEST_DIRS
    }


def policed_cuda_targets() -> list[Path]:
    """Every CUDA test target the source scan is responsible for.

    Both scans and the census enumerate through here rather than each globbing
    for themselves. A census that walked its own copy of this loop could
    disagree with the scans about what is in scope, and would then be
    reporting on a set nobody checks -- so the enumeration is shared to make
    that divergence unrepresentable rather than merely unlikely.
    """
    return [
        path for targets in policed_cuda_targets_by_dir().values() for path in targets
    ]


# `ignore` as an attribute token, not as English. The first version of this
# scan searched the item's head for the substring "ignore", and so passed on a
# test whose doc comment read "a kernel that ignored the attribute" -- the exact
# defect it was written to catch. Anchor on the punctuation an attribute must
# have around it, and read only masked attribute text, never prose or literals.
IGNORE_ATTR = re.compile(r"(?:^|[\[(,\s])ignore\s*(?:=|\]|,|\)|$)")
# An ignore inside a `cfg_attr` only ignores the test when its predicate holds.
# `#[cfg_attr(not(feature = "some-other-thing"), ignore = "...")]` still runs
# the test with `gpu-tests` off, which is the defect, so the predicate has to be
# read rather than assumed from the presence of an `ignore` token.
GPU_TESTS_PREDICATE = re.compile(r"""\s*not\s*\(\s*feature\s*=\s*["']gpu-tests["']\s*\)\s*""")
RAW_STRING_START = re.compile(r"r(#*)\"")
CHAR_LITERAL = re.compile(r"'(?:\\.|[^\\'])'")
# Walking upward out of one item and into the one above it would let a test
# inherit its neighbour's ignore -- a false negative in a check that must fail
# closed. These anchor the top of an item's head regardless of bracket counting.
ITEM_BOUNDARY = re.compile(
    r"^(?:\}|\{|fn\s|pub\s|async\s|unsafe\s|extern\s|mod\s|use\s|let\s|impl\s"
    r"|struct\s|enum\s|trait\s|const\s|static\s|type\s|macro_rules)"
)
# `pub(crate) fn neighbour()` is an item opener that `pub\s` above does not
# match, and a walk that runs past it adopts the neighbour's ignore. Anchoring
# on the `fn` token itself covers every visibility spelling at once, rather
# than enumerating them and being one spelling short again later.
FN_DECLARATION = re.compile(r"\bfn\s+[A-Za-z0-9_]+\s*[(<]")
# `#[test]`, `#[ test ]`, and the same-line `#[test] fn a() {}` all produce a
# running test. Matching only the canonical spelling left two of them uninspected.
TEST_ATTR = re.compile(r"#\[\s*test\s*\]")
# Only `gpu-tests` distinguishes the two configurations this script builds
# (`cuda` and `cuda,gpu-tests`), and it is a leaf feature -- ep-cuda forwards it
# to cuda-memory, where it is `[]`. So a `cfg` on any *other* feature is held
# constant across both builds, produces no inventory drift, and `compare_inventories`
# passes it. Flagging those would red the fast lane on code the authoritative
# check accepts, which is worse than missing one: `#![cfg(feature = "cuda")]`
# already exists in this tree, on an unpoliced target.
GPU_TESTS_FEATURE = "gpu-tests"
# A `cfg` naming a feature *deletes* code from one configuration rather than
# ignoring it, so the two inventories stop matching. `cfg_attr` is a different
# attribute and is the sanctioned spelling -- `cfg\s*\(` cannot match
# `cfg_attr(` because `_` is not `(`, so the two are distinguished by shape.
FEATURE_CFG = re.compile(r"cfg\s*\(")
FEATURE_MENTION = re.compile(r"\bfeature\s*=")
# Only the feature name is quoted, so it survives in raw text where masking
# blanks it; detection runs on masked text, quoting on raw, as everywhere else.
FEATURE_NAME = re.compile(r"""feature\s*=\s*["']([A-Za-z0-9_-]+)["']""")
# A `mod` gated on `gpu-tests` removes every test inside it from one inventory,
# which is the same drift one level up from a test's own attributes.
MOD_DECLARATION = re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+[A-Za-z0-9_]+\s*\{")


def mask_source(source: str) -> list[tuple[str, bool]]:
    """Per line: (code with literals and comments masked, is-comment-only).

    Length-preserving and column-exact, so a span found in the masked text can
    be read back out of the original. Bracket counting and token matching both
    run on the masked text: a `)` inside `expected = "boom )"` is not structure,
    and `ignore` inside a string or a doc comment is not an attribute. Counting
    them was a fail-open -- one unbalanced bracket in a string literal let the
    walk escape upward and adopt the previous test's ignore.
    """
    masked: list[tuple[str, bool]] = []
    block_depth = 0
    raw_hashes: int | None = None
    in_string = False
    for line in source.splitlines():
        code: list[str] = []
        index, length = 0, len(line)
        while index < length:
            if block_depth:
                if line.startswith("*/", index):
                    block_depth -= 1
                    code.append("  ")
                    index += 2
                elif line.startswith("/*", index):
                    block_depth += 1
                    code.append("  ")
                    index += 2
                else:
                    code.append(" ")
                    index += 1
                continue
            if raw_hashes is not None:
                terminator = '"' + "#" * raw_hashes
                at = line.find(terminator, index)
                if at == -1:
                    code.append("_" * (length - index))
                    index = length
                else:
                    code.append("_" * (at - index))
                    code.append('"' + " " * raw_hashes)
                    index = at + len(terminator)
                    raw_hashes = None
                continue
            if in_string:
                cursor = index
                while cursor < length:
                    if line[cursor] == "\\":
                        cursor += 2
                        continue
                    if line[cursor] == '"':
                        break
                    cursor += 1
                if cursor >= length:
                    code.append("_" * (length - index))
                    index = length
                else:
                    code.append("_" * (cursor - index))
                    code.append('"')
                    index = cursor + 1
                    in_string = False
                continue
            if line.startswith("//", index):
                code.append(" " * (length - index))
                index = length
                continue
            if line.startswith("/*", index):
                block_depth = 1
                code.append("  ")
                index += 2
                continue
            raw = RAW_STRING_START.match(line, index)
            if raw:
                raw_hashes = len(raw.group(1))
                code.append(" " * (raw.end() - index - 1) + '"')
                index = raw.end()
                continue
            char = CHAR_LITERAL.match(line, index)
            if char:
                code.append("'" + "_" * (char.end() - index - 2) + "'")
                index = char.end()
                continue
            if line[index] == '"':
                in_string = True
                code.append('"')
                index += 1
                continue
            code.append(line[index])
            index += 1
        rendered = "".join(code)
        masked.append((rendered, bool(line.strip()) and not rendered.strip()))
    return masked


def head_line_indices(masked: Sequence[tuple[str, bool]], index: int, downward: bool = True) -> list[int]:
    """Line indices of the attribute head of the `#[test]` item at `index`."""
    above: list[int] = []
    pending = 0
    for cursor in range(index - 1, -1, -1):
        code, comment_only = masked[cursor]
        stripped = code.strip()
        # Gated on depth, like the downward walk: a blank line inside a
        # multi-line attribute is legal and must not truncate the head, or a
        # correctly ignored test gets reported and reds an otherwise green lane.
        if pending == 0 and not stripped and not comment_only:
            break
        # Checked regardless of bracket depth, on purpose: this is the backstop
        # that makes the walk degrade closed if bracket accounting is ever
        # wrong. After masking, a continuation line cannot look like the start
        # of an item, because anything that could is inside a literal.
        if TEST_ATTR.search(stripped) or ITEM_BOUNDARY.match(stripped) or FN_DECLARATION.search(stripped):
            break
        if pending == 0 and comment_only:
            continue
        delta = stripped.count("]") + stripped.count(")") - stripped.count("[") - stripped.count("(")
        # A multi-line attribute is met tail-first walking upward, so a line
        # closing more brackets than it opens starts a continuation.
        if pending == 0 and delta <= 0 and not stripped.startswith("#["):
            break
        above.append(cursor)
        pending = max(0, pending + delta)
    head = list(reversed(above))
    if not downward:
        return head
    pending = 0
    for cursor in range(index + 1, len(masked)):
        code, comment_only = masked[cursor]
        stripped = code.strip()
        if pending == 0:
            if comment_only:
                continue
            if not stripped.startswith("#["):
                break
        head.append(cursor)
        pending = max(0, pending + stripped.count("[") + stripped.count("(") - stripped.count("]") - stripped.count(")"))
    return head


def attribute_spans(text: str, opener: str = "#[") -> list[tuple[int, int]]:
    """Half-open spans of each `#[..]` attribute in bracket-masked `text`.

    `opener` selects outer (`#[`) or inner (`#![`) attributes. The bracket walk
    is identical for both -- it skips forward to the first bracket either way --
    so the two share one definition rather than one being reimplemented later
    and drifting from the masking rules the other depends on.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(opener, cursor)
        if start == -1:
            return spans
        depth = 0
        end = start + 1
        while end < len(text):
            char = text[end]
            if char in "[(":
                depth += 1
            elif char in "])":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if depth != 0:
            return spans
        spans.append((start, end + 1))
        cursor = end + 1


def cfg_attr_ignores_without_gpu_tests(masked_attribute: str, raw_attribute: str) -> bool:
    """Whether a `cfg_attr` ignores the test exactly when `gpu-tests` is off.

    Reading the predicate as a substring is not enough. `all(not(feature =
    "gpu-tests"), windows)` contains it and still runs the test on the Linux
    lane, and a nested `cfg_attr` can gate the ignore on something else
    entirely. The predicate has to be the whole predicate, and the ignore has
    to be the outer attribute's own.
    """
    opener = re.search(r"cfg_attr\s*\(", masked_attribute)
    if not opener:
        return False
    depth, cursor, comma = 1, opener.end(), None
    while cursor < len(masked_attribute):
        char = masked_attribute[cursor]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1 and comma is None:
            comma = cursor
        cursor += 1
    if comma is None:
        return False
    if "cfg_attr" in masked_attribute[comma:cursor]:
        return False
    if not IGNORE_ATTR.search("," + masked_attribute[comma + 1 : cursor]):
        return False
    return GPU_TESTS_PREDICATE.fullmatch(raw_attribute[opener.end() : comma]) is not None


def has_effective_ignore(masked_head: str, raw_head: str) -> bool:
    """Whether the head ignores the test when `gpu-tests` is off.

    The ignore token is looked for in masked text so literals and prose cannot
    supply it; the `cfg_attr` predicate is read from the original text at the
    same offsets, because masking blanks the feature name it needs to check.
    """
    for start, end in attribute_spans(masked_head):
        attribute = masked_head[start:end]
        if not IGNORE_ATTR.search(attribute):
            continue
        if "cfg_attr" not in attribute:
            return True
        if cfg_attr_ignores_without_gpu_tests(attribute, raw_head[start:end]):
            return True
    return False


def test_name_at(lines: list[str], index: int) -> str:
    for line in lines[index + 1 : index + 12]:
        match = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)", line.strip())
        if match:
            return match.group(1)
    return "<unknown fn>"


@dataclass(frozen=True)
class TestItem:
    """A `#[test]` item's head: its attributes, in both masked and raw form."""

    line: int
    name: str
    masked_head: str
    raw_head: str
    aligned: bool


@lru_cache(maxsize=None)
def masked_lines(source: str) -> tuple[tuple[str, bool], ...]:
    """`mask_source` memoised: both source scans mask the same files."""
    return tuple(mask_source(source))


def test_items(source: str) -> list[TestItem]:
    """Every `#[test]` item in `source`, with its full attribute head.

    The upward/downward walk is subtle enough that having two copies of it
    would guarantee they eventually disagree, so both source scans read the
    item head from here.
    """
    lines = source.splitlines()
    masked = masked_lines(source)
    items: list[TestItem] = []
    for index, (code, _) in enumerate(masked):
        spans = attribute_spans(code)
        if not any(TEST_ATTR.fullmatch(code[start:end].strip()) for start, end in spans):
            continue
        # `#[test] fn a() {}` puts the whole item on one line, so walking
        # downward would collect the *next* item's attributes and let this test
        # inherit them. The line's own attributes are part of the head either way.
        same_line_fn = FN_DECLARATION.search(code)
        head = head_line_indices(masked, index, downward=not same_line_fn)
        # Joined unstripped: masking is length-preserving per line, so the two
        # blobs agree column for column and a span found in one can be read out
        # of the other. Stripping would break exactly that, because masking a
        # trailing comment leaves spaces that strip() then removes.
        masked_head = "\n".join([masked[cursor][0] for cursor in head] + [code])
        raw_head = "\n".join([lines[cursor] for cursor in head] + [lines[index]])
        name = same_line_fn.group(0) if same_line_fn else None
        items.append(
            TestItem(
                line=index + 1,
                name=name[3:].strip("(<").strip() if name else test_name_at(lines, index),
                masked_head=masked_head,
                raw_head=raw_head,
                # Masking is column-exact, so the two blobs cannot differ in
                # length; if they ever do, the span mapping has drifted and
                # every offset read out of it is untrustworthy.
                aligned=len(masked_head) == len(raw_head),
            )
        )
    return items


def function_body(source: str, name: str, start_line: int = 1) -> str | None:
    """Masked body of `name` at or below `start_line`, or None if malformed."""
    blobs = source_blobs(source)
    if blobs is None:
        return None
    masked_blob, _ = blobs
    lines = masked_blob.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: max(0, start_line - 1)])
    declaration = re.search(rf"\bfn\s+{re.escape(name)}\s*[(<]", masked_blob[start:])
    if declaration is None:
        return None
    open_brace = masked_blob.find("{", start + declaration.end())
    if open_brace == -1:
        return None
    depth = 0
    for cursor in range(open_brace, len(masked_blob)):
        if masked_blob[cursor] == "{":
            depth += 1
        elif masked_blob[cursor] == "}":
            depth -= 1
            if depth == 0:
                return masked_blob[open_brace + 1 : cursor]
    return None


def tests_missing_suite_lock(source: str, lock_function: str) -> list[TestItem]:
    """Tests that do not hold the target's suite lock through function exit.

    The accepted shape deliberately binds the guard as the first statement and
    never names that binding again. A Rust local otherwise lives to the end of
    its enclosing function, so this proves the critical section includes every
    later operation and assertion. Any later identifier use fails closed: it
    covers explicit/qualified `drop`, assignment, shadowing, and moving the
    guard into a call or shorter scope without pretending this source guard is
    a complete Rust parser. Comments and literals are already blanked by
    `function_body`, so prose cannot manufacture a premature release.
    """
    first_statement = re.compile(
        rf"\s*let\s+(?:mut\s+)?(?P<guard>[A-Za-z][A-Za-z0-9_]*|_[A-Za-z0-9_]+)\s*=\s*"
        rf"{re.escape(lock_function)}\s*\(\s*\)\s*;"
    )
    missing: list[TestItem] = []
    for item in test_items(source):
        body = function_body(source, item.name, item.line)
        acquisition = first_statement.match(body) if body is not None else None
        if acquisition is None:
            missing.append(item)
            continue
        guard = acquisition.group("guard")
        later_guard_use = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(guard)}(?![A-Za-z0-9_])",
            body[acquisition.end() :],
        )
        if later_guard_use is not None:
            missing.append(item)
    return missing


def suite_isolation_errors(
    configured: Mapping[str, tuple[str, str]],
    sources: Mapping[str, str],
    expected: frozenset[str] = EXPECTED_SUITE_ISOLATION_TARGETS,
) -> list[str]:
    """Validate the independent target census and every configured source."""
    errors: list[str] = []

    if not expected:
        return [
            "suite-isolation policy target census is empty; the guard refuses to pass vacuously"
        ]
    if not configured:
        return [
            "suite-isolation lock configuration is empty; expected exactly "
            + ", ".join(sorted(expected))
        ]

    actual = frozenset(configured)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        errors.append(
            "suite-isolation configured target census does not match policy: "
            + "; ".join(details)
        )

    for target in sorted(expected & actual):
        lock_function, lock_static = configured[target]
        relative = Path("crates") / "onnx-runtime-ep-cuda" / "tests" / f"{target}.rs"
        source = sources.get(target)
        if source is None:
            errors.append(
                f"{relative}: serialized CUDA target is missing; the suite-isolation "
                "census cannot verify it"
            )
            continue
        blobs = source_blobs(source)
        if blobs is None:
            errors.append(
                f"{relative}: source masking changed length; refusing to trust the "
                "suite-isolation scan"
            )
            continue
        masked, _ = blobs
        items = test_items(source)
        if not items:
            errors.append(
                f"{relative}: serialized CUDA target contains no #[test] items; "
                "suite isolation would otherwise pass vacuously"
            )
        static_declaration = re.compile(
            rf"\bstatic\s+{re.escape(lock_static)}\s*:\s*Mutex\s*<\s*\(\s*\)\s*>\s*="
        )
        helper = function_body(source, lock_function)
        if (
            static_declaration.search(masked) is None
            or helper is None
            or re.search(rf"\b{re.escape(lock_static)}\s*\.\s*lock\s*\(\s*\)", helper)
            is None
        ):
            errors.append(
                f"{relative}: {lock_function} must acquire the process-wide "
                f"{lock_static}; otherwise per-test calls do not serialize the target"
            )
        for item in tests_missing_suite_lock(source, lock_function):
            errors.append(
                f"{relative}:{item.line}: {item.name} must acquire {lock_function}() "
                "as its first statement and keep that guard live through function exit "
                "so EP construction, execution, telemetry, and assertions share one "
                "critical section"
            )
    return errors


def scan_source_for_suite_isolation() -> list[str]:
    """Affected CUDA targets lock before EP construction, execution, or stats."""
    sources: dict[str, str] = {}
    for target in EXPECTED_SUITE_ISOLATION_TARGETS:
        path = TESTS / f"{target}.rs"
        if path.is_file():
            sources[target] = path.read_text(encoding="utf-8")
    return suite_isolation_errors(SERIALIZED_TARGET_LOCKS, sources)


def un_ignored_tests(source: str) -> list[tuple[int, str]]:
    """Every `#[test]` in `source` not ignored when `gpu-tests` is off, 1-based."""
    found: list[tuple[int, str]] = []
    for item in test_items(source):
        if not item.aligned:
            found.append((item.line, item.name))
            continue
        if has_effective_ignore(item.masked_head, item.raw_head):
            continue
        found.append((item.line, item.name))
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
    for path in policed_cuda_targets():
        source = path.read_text(encoding="utf-8")
        for line_number, name in un_ignored_tests(source):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: {name} "
                "has no ignore attribute. Every test in a CUDA target must be ignored "
                "when gpu-tests is off, or it runs on a machine with no device: add "
                '#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]'
            )
    return errors


def gates_on_gpu_tests(masked: str, raw: str, start: int, end: int) -> bool:
    """Whether the attribute at `start:end` is a `cfg` on `gpu-tests`.

    Detection reads masked text so a `cfg` written in a doc comment or a string
    cannot trigger it; the feature names are quoted out of raw text at the same
    offsets, because masking blanks the string literals that hold them.

    Only `gpu-tests` counts. A `cfg` on `cuda` or `tracing` resolves the same
    way in both builds, so it cannot move a test between the two inventories,
    and flagging it would fail a lane the authoritative check passes.
    """
    attribute = masked[start:end]
    if not FEATURE_CFG.search(attribute) or not FEATURE_MENTION.search(attribute):
        return False
    return GPU_TESTS_FEATURE in FEATURE_NAME.findall(raw[start:end])


def feature_gated_attribute(masked: str, raw: str, opener: str = "#[") -> bool:
    """Whether any `cfg` attribute in `masked` gates on `gpu-tests`."""
    return any(gates_on_gpu_tests(masked, raw, start, end) for start, end in attribute_spans(masked, opener))


def source_blobs(source: str) -> tuple[str, str] | None:
    """The whole file as masked and raw blobs, or None if they ever disagree."""
    lines = source.splitlines()
    masked_blob = "\n".join(code for code, _ in masked_lines(source))
    raw_blob = "\n".join(lines)
    return None if len(masked_blob) != len(raw_blob) else (masked_blob, raw_blob)


def target_level_feature_gate(source: str) -> int | None:
    """The 1-based line of a `#![cfg(feature = "gpu-tests")]` on the target."""
    blobs = source_blobs(source)
    if blobs is None:
        return None
    masked_blob, raw_blob = blobs
    for start, end in attribute_spans(masked_blob, "#!["):
        if gates_on_gpu_tests(masked_blob, raw_blob, start, end):
            return masked_blob.count("\n", 0, start) + 1
    return None


def gpu_tests_gated_modules(source: str) -> list[int]:
    """1-based lines of `mod` items gated on `gpu-tests` that contain tests.

    A test's own attributes are not the only way to delete it from the base
    inventory -- a `#[cfg(feature = "gpu-tests")]` on the module around it does
    the same thing, and the head walk deliberately stops at the `mod` boundary,
    so it is invisible from inside. Only modules that actually contain a test
    are reported: gating a helper module is the sanctioned two-arm shim.
    """
    blobs = source_blobs(source)
    if blobs is None:
        return []
    masked_blob, raw_blob = blobs
    masked = masked_lines(source)
    found: list[int] = []
    offsets, cursor = [], 0
    for code, _ in masked:
        offsets.append(cursor)
        cursor += len(code) + 1
    for index, (code, _) in enumerate(masked):
        if not MOD_DECLARATION.match(code):
            continue
        head = head_line_indices(masked, index, downward=False)
        if not head:
            continue
        head_start, head_end = offsets[head[0]], offsets[index]
        gate = next(
            (
                head_start + start
                for start, end in attribute_spans(masked_blob[head_start:head_end])
                if gates_on_gpu_tests(masked_blob[head_start:head_end], raw_blob[head_start:head_end], start, end)
            ),
            None,
        )
        if gate is None:
            continue
        body_start = offsets[index] + code.index("{")
        depth, cursor = 0, body_start
        while cursor < len(masked_blob):
            if masked_blob[cursor] == "{":
                depth += 1
            elif masked_blob[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        if TEST_ATTR.search(masked_blob[body_start:cursor]):
            found.append(masked_blob.count("\n", 0, gate) + 1)
    return found


def manifest_required_features(manifest: str) -> Iterator[str]:
    """Policed CUDA targets a manifest gives `required-features`.

    Parsed rather than pattern-matched. A regex over `[[test]]` sections was
    wrong in both directions: it absorbed a following `[[bench]]` table into
    the preceding test's span, and it did not recognise a header with a
    trailing comment. Neither shape is exotic, and both are free to get right.
    """
    for section in tomllib.loads(manifest).get("test", []):
        name = section.get("name")
        if isinstance(name, str) and is_cuda_test_target(name) and section.get("required-features"):
            yield name


def targets_with_required_features() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for crate in CUDA_CRATES:
        manifest_path = crate / "Cargo.toml"
        if not manifest_path.is_file():
            continue
        found.extend(
            (manifest_path, name)
            for name in manifest_required_features(manifest_path.read_text(encoding="utf-8"))
        )
    return found


def scan_source_for_inventory_drift() -> list[str]:
    """No policed CUDA target may hold fewer tests with `gpu-tests` off than on.

    A missing ignore makes a test *run* where there is no device. This is the
    other half of the same class: a `cfg` on a feature makes a test, or a whole
    target, *cease to exist* in the configuration CI builds. Both break the
    lane, but this half is the quieter one -- the target compiles, the suite is
    green, and it is green because the tests are not there. `#1854` shipped a
    target-level gate that took a 13-test file to 0 with `gpu-tests` off.

    `compare_inventories` is authoritative for this and needs both CUDA builds
    to say so. These three spellings are the ones that have actually occurred,
    and all three are visible in the source.
    """
    errors: list[str] = []
    for path in policed_cuda_targets():
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        gate = target_level_feature_gate(source)
        if gate is not None:
            errors.append(
                f'{relative}:{gate}: the whole target is gated on feature "{GPU_TESTS_FEATURE}". '
                "With the feature off the target holds no tests at all, so it cannot disagree "
                "with the device suite and cannot go red. Gate the item that needs the feature "
                "-- a two-arm shim over the gated helper keeps every test in both inventories."
            )
        for line_number in gpu_tests_gated_modules(source):
            errors.append(
                f'{relative}:{line_number}: this module is gated on feature "{GPU_TESTS_FEATURE}" '
                "and contains tests, so those tests are absent from the inventory when the "
                "feature is off. Gate the helper the module exists for, not the tests."
            )
        for item in test_items(source):
            if not item.aligned:
                # Masking is column-exact, so this cannot happen; if it
                # ever does, report rather than trust a span mapping that
                # has drifted -- the same disposition as `un_ignored_tests`.
                errors.append(
                    f"{relative}:{item.line}: {item.name} could not be parsed reliably "
                    "(masked and raw text disagree in length); refusing to clear it."
                )
                continue
            if not feature_gated_attribute(item.masked_head, item.raw_head):
                continue
            errors.append(
                f'{relative}:{item.line}: {item.name} is gated on feature "{GPU_TESTS_FEATURE}", '
                "so it is absent from the inventory when the feature is off. Use "
                '#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")] '
                "to keep the test present and skipped rather than deleted."
            )
    for manifest_path, target in targets_with_required_features():
        errors.append(
            f"{manifest_path.relative_to(ROOT)}: test target {target} sets required-features, "
            "so it is not built at all in the base configuration and its tests are missing from "
            "that inventory. Build the target unconditionally and ignore its tests instead."
        )
    return errors


def census_errors(by_dir: Mapping[Path, Sequence[Path]]) -> list[str]:
    """Every policed directory must have contributed something to scan.

    Both source scans report per-target, so a directory with no targets
    produces no findings and reads as clean -- which is the second shape of
    the #1875 defect (tests leave the inventory, the lane goes green because
    the tests are not there) at directory granularity. A guard keyed on the
    scans' findings cannot see it, because the fault removes the things the
    guard reads.

    The check is per-directory rather than on the total. An aggregate
    non-empty test only defends the all-or-nothing case: if one directory has
    many targets and another has fewer, deleting either can leave the total
    comfortably non-zero and the gate reports a confident pass over the
    remainder. The crates have already split once for exactly this kind of
    reason, so a move that lands one directory somewhere TEST_DIRS does not
    point at is the realistic version of this fault, not the exotic one.

    A missing directory contributes nothing and so fails here too, which is
    the intent: TEST_DIRS is a constant in this file and the crate layout
    around it is not, so the constant going stale must be loud.
    """
    errors: list[str] = []
    for test_dir, targets in by_dir.items():
        if targets:
            continue
        errors.append(
            f"{test_dir.relative_to(ROOT)} contributed no CUDA test targets to the scan. "
            "Either it was deleted, or the crate layout moved and TEST_DIRS in this script "
            "is now pointing at nothing. Every per-target check is vacuous for this "
            "directory until that is resolved."
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


def parse_named_test_binary_from_json(stdout: str, expected_target: str) -> TestBinary | None:
    """Find one explicitly requested integration-test artifact in Cargo JSON."""
    binary: TestBinary | None = None
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message.get("target", {})
        if target.get("name") != expected_target or "test" not in target.get("kind", []):
            continue
        executable = message.get("executable")
        if executable:
            binary = TestBinary(expected_target, Path(executable))
    return binary


def feature_target_inventory_errors(
    label: str,
    binary: TestBinary | None,
    inventory: frozenset[str],
    minimum: int,
) -> list[str]:
    errors: list[str] = []
    if binary is None:
        errors.append(
            f"{label}: Cargo emitted no executable for the requested feature-gated test target"
        )
    if len(inventory) < minimum:
        errors.append(
            f"{label}: feature-enabled inventory has {len(inventory)} test(s); "
            f"expected at least {minimum}"
        )
    return errors


def build_named_test_binary(
    package: str, target: str, features: str
) -> TestBinary:
    result = run(
        [
            "cargo",
            "test",
            "--locked",
            "-p",
            package,
            "--no-default-features",
            "--features",
            features,
            "--test",
            target,
            "--no-run",
            "--message-format=json",
        ]
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr, end="")
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"cargo test --no-run failed for {package}/{target} with {features}"
        )
    binary = parse_named_test_binary_from_json(result.stdout, target)
    errors = feature_target_inventory_errors(
        f"{package}/{target}", binary, frozenset(), 0
    )
    if errors:
        raise RuntimeError(errors[0])
    assert binary is not None
    return binary


def verify_production_feature_targets() -> tuple[list[str], list[str]]:
    """Compile and census production GPU tests without executing GPU paths."""
    errors: list[str] = []

    plugin_binary = build_named_test_binary(
        CUDA_PLUGIN_PACKAGE, CUDA_PLUGIN_TARGET, CUDA_PLUGIN_FEATURES
    )
    plugin_inventory = list_inventory(plugin_binary)
    errors.extend(
        feature_target_inventory_errors(
            f"{CUDA_PLUGIN_PACKAGE}/{CUDA_PLUGIN_TARGET} with {CUDA_PLUGIN_FEATURES}",
            plugin_binary,
            plugin_inventory,
            CUDA_PLUGIN_MIN_TESTS,
        )
    )

    session_base_binary = build_named_test_binary(
        SESSION_PACKAGE, SESSION_HETERO_TARGET, SESSION_BASE_FEATURES
    )
    session_base_inventory = list_inventory(session_base_binary)
    errors.extend(
        feature_target_inventory_errors(
            f"{SESSION_PACKAGE}/{SESSION_HETERO_TARGET} without gpu-tests",
            session_base_binary,
            session_base_inventory,
            SESSION_HETERO_MIN_TESTS,
        )
    )
    _, passed, failed, ignored, names = run_libtest(session_base_binary)
    errors.extend(
        validate_ignored_result(
            IgnoredResult(
                SESSION_HETERO_TARGET,
                len(session_base_inventory),
                passed,
                failed,
                ignored,
                known_names(names, session_base_inventory),
            )
        )
    )

    session_gpu_binary = build_named_test_binary(
        SESSION_PACKAGE, SESSION_HETERO_TARGET, SESSION_GPU_FEATURES
    )
    session_gpu_inventory = list_inventory(session_gpu_binary)
    errors.extend(
        feature_target_inventory_errors(
            f"{SESSION_PACKAGE}/{SESSION_HETERO_TARGET} with gpu-tests",
            session_gpu_binary,
            session_gpu_inventory,
            SESSION_HETERO_MIN_TESTS,
        )
    )
    errors.extend(
        compare_inventories(
            {SESSION_HETERO_TARGET: session_base_inventory},
            {SESSION_HETERO_TARGET: session_gpu_inventory},
        )
    )

    engine_base_binary = build_named_test_binary(
        ENGINE_PACKAGE, ENGINE_DEFAULT_OFF_TARGET, ENGINE_BASE_FEATURES
    )
    engine_base_inventory = list_inventory(engine_base_binary)
    errors.extend(
        feature_target_inventory_errors(
            f"{ENGINE_PACKAGE}/{ENGINE_DEFAULT_OFF_TARGET} without gpu-tests",
            engine_base_binary,
            engine_base_inventory,
            ENGINE_DEFAULT_OFF_MIN_TESTS,
        )
    )
    _, passed, failed, ignored, names = run_libtest(engine_base_binary)
    errors.extend(
        validate_ignored_result(
            IgnoredResult(
                ENGINE_DEFAULT_OFF_TARGET,
                len(engine_base_inventory),
                passed,
                failed,
                ignored,
                known_names(names, engine_base_inventory),
            )
        )
    )

    engine_gpu_binary = build_named_test_binary(
        ENGINE_PACKAGE, ENGINE_DEFAULT_OFF_TARGET, ENGINE_GPU_FEATURES
    )
    engine_gpu_inventory = list_inventory(engine_gpu_binary)
    errors.extend(
        feature_target_inventory_errors(
            f"{ENGINE_PACKAGE}/{ENGINE_DEFAULT_OFF_TARGET} with gpu-tests",
            engine_gpu_binary,
            engine_gpu_inventory,
            ENGINE_DEFAULT_OFF_MIN_TESTS,
        )
    )
    errors.extend(
        compare_inventories(
            {ENGINE_DEFAULT_OFF_TARGET: engine_base_inventory},
            {ENGINE_DEFAULT_OFF_TARGET: engine_gpu_inventory},
        )
    )

    summaries = [
        f"{CUDA_PLUGIN_PACKAGE}/{CUDA_PLUGIN_TARGET}: "
        f"{len(plugin_inventory)} test(s) compiled with {CUDA_PLUGIN_FEATURES}",
        f"{SESSION_PACKAGE}/{SESSION_HETERO_TARGET}: "
        f"{len(session_base_inventory)} test(s) present without gpu-tests and ignored; "
        f"{len(session_gpu_inventory)} present with gpu-tests and compiled only",
        f"{ENGINE_PACKAGE}/{ENGINE_DEFAULT_OFF_TARGET}: "
        f"{len(engine_base_inventory)} test(s) present without gpu-tests and ignored; "
        f"{len(engine_gpu_inventory)} present with gpu-tests and compiled only",
    ]
    return summaries, errors


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
    if re.search(r"\b\d+\s+targets?\b", census_errors.__doc__ or ""):
        raise AssertionError(
            "census_errors documentation must describe relative inventory sizes, "
            "not copy a source-derived target count that will go stale"
        )
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
    named_binary = parse_named_test_binary_from_json(fixture_stdout, "fixture_gpu")
    if named_binary != expected[0]:
        raise AssertionError(f"named feature target parser returned {named_binary!r}")
    missing_target_errors = feature_target_inventory_errors(
        "missing-feature-target", None, frozenset(), 1
    )
    if not any("no executable" in error for error in missing_target_errors):
        raise AssertionError(
            "a missing feature-gated target must fail even before its inventory is read"
        )
    empty_feature_errors = feature_target_inventory_errors(
        "empty-feature-target", expected[0], frozenset(), 1
    )
    if not any("inventory has 0" in error for error in empty_feature_errors):
        raise AssertionError("an empty feature-enabled target inventory must fail")

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

    # These fixtures are the review history of this scan, kept as tests. The
    # first version searched the item head for the substring "ignore" and so
    # accepted a test whose doc comment said "a kernel that ignored the
    # attribute" -- the defect it exists to catch. The second version counted
    # brackets in string literals, so one stray `)` inside a `should_panic`
    # message let the upward walk escape into the item above and adopt *its*
    # ignore: a fail-open in a check that must fail closed. Both controls are
    # here on purpose -- the cases that must be flagged, and the ones that must
    # not, because only the second kind catches a scan that flags everything.
    for source, expected in (
        ("#[test]\nfn a() {}\n", ["a"]),
        ("#[ignore]\n#[test]\nfn a() {}\n", []),
        ('#[test]\n#[ignore = "needs a device"]\nfn a() {}\n', []),
        ('#[cfg_attr(not(feature = "gpu-tests"), ignore = "x")]\n#[test]\nfn a() {}\n', []),
        ('#[cfg_attr(not(feature = "gpu-tests"), ignore)]\n#[test]\nfn a() {}\n', []),
        ('#[cfg_attr(\n    not(feature = "gpu-tests"),\n    ignore = "x"\n)]\n#[test]\nfn a() {}\n', []),
        ('#[cfg(feature = "gpu-tests")]\n#[test]\nfn a() {}\n', ["a"]),
        # An ignore conditioned on some other feature does not ignore the test
        # when gpu-tests is off, which is the whole defect.
        ('#[cfg_attr(not(feature = "something-else"), ignore = "x")]\n#[test]\nfn a() {}\n', ["a"]),
        # Prose is not an attribute: both of these say "ignore" and neither is one.
        ("/// a kernel that ignored the attribute entirely\n#[test]\nfn a() {}\n", ["a"]),
        ("/// we do not ignore, ever\n#[test]\nfn a() {}\n", ["a"]),
        ("// let ignore = 1;\n#[test]\nfn a() {}\n", ["a"]),
        # Nor is a string literal, however attribute-shaped its contents.
        ('#[should_panic(expected = "ignore me")]\n#[test]\nfn a() {}\n', ["a"]),
        ("#[allow(clippy::ignored_unit_patterns)]\n#[test]\nfn a() {}\n", ["a"]),
        # A bracket inside a literal is not structure. Without masking, the
        # stray `)` below makes the walk climb into `a` and inherit its ignore.
        (
            '#[cfg_attr(not(feature = "gpu-tests"), ignore = "x")]\n#[test]\nfn a() {}\n'
            '#[should_panic(expected = "boom )")]\n#[test]\nfn b() {}\n',
            ["b"],
        ),
        ('/// we do not ignore, ever\n#[should_panic(expected = "boom )")]\n#[test]\nfn a() {}\n', ["a"]),
        ('#[cfg_attr(not(feature = "gpu-tests"), ignore = r#"a ) ] b"#)]\n#[test]\nfn a() {}\n', []),
        # Comments inside a head must not truncate it, including nested blocks.
        ('#[cfg_attr(not(feature = "gpu-tests"), ignore = "x")]\n/* why */\n#[test]\nfn a() {}\n', []),
        ('#[cfg_attr(not(feature = "gpu-tests"), ignore = "x")]\n/* a /* b */ c */\n#[test]\nfn a() {}\n', []),
        ('/// doc\n#[cfg_attr(not(feature = "gpu-tests"), ignore = "x")]\n#[test]\nfn a() {}\n', []),
        # An adjacent item's ignore is not this item's, blank line or not.
        ("#[test]\nfn a() {}\n\n#[ignore]\n#[test]\nfn b() {}\n", ["a"]),
        ("#[ignore]\n#[test]\nfn a() {}\n#[test]\nfn b() {}\n", ["b"]),
        ("macro_rules! m {\n    () => {\n        #[ignore]\n        #[test]\n        fn a() {}\n    };\n}\n", []),
        # A second review found these four, all of them residual holes in the
        # first fix. Three were fail-open; the first would have reddened a green
        # lane on correct code, which is worse than useless for a pre-merge gate.
        #
        # A blank line inside a multi-line attribute is legal, and truncating
        # the head there reports a correctly ignored test.
        ('#[cfg_attr(\n\n    not(feature = "gpu-tests"),\n    ignore = "x"\n)]\n#[test]\nfn a() {}\n', []),
        # The predicate must be the whole predicate. Both of these contain
        # `not(feature = "gpu-tests")` as a substring and both still run the
        # test on the Linux CUDA lane with gpu-tests off.
        ('#[cfg_attr(all(not(feature = "gpu-tests"), windows), ignore)]\n#[test]\nfn a() {}\n', ["a"]),
        ('#[cfg_attr(not(feature = "gpu-tests"), cfg_attr(feature = "foo", ignore))]\n#[test]\nfn a() {}\n', ["a"]),
        # `#[test]` is not always spelled `#[test]` on a line of its own. Both
        # of these compile and run; matching only the canonical form never even
        # looked at them.
        ("#[test] fn foo() {}\n", ["foo"]),
        ("#[ test ]\nfn spaced() {}\n", ["spaced"]),
        ("#[ignore] #[test] fn foo() {}\n", []),
        # ...and with the item on one line, walking downward would collect the
        # *next* item's attributes and inherit its ignore.
        ("#[test] fn foo() {}\n#[ignore]\n#[test]\nfn bar() {}\n", ["foo"]),
        # `pub(crate) fn` is an item opener that a `pub\s` boundary misses, and
        # the walk then climbs into the neighbour above and adopts its ignore.
        ("#[test]\n#[ignore]\npub(crate) fn neighbour() { assert_eq!(\n    1, 1,\n); }\n#[test]\nfn ours() {}\n", ["ours"]),
        # An attribute below `#[test]` with a trailing comment. Masking that
        # comment leaves trailing spaces, so joining the stripped lines made the
        # masked and raw blobs different lengths and the span mapping drifted.
        (
            '#[cfg_attr(\n    not(feature = "gpu-tests"),\n    ignore = "x"\n)]\n#[test]\n'
            "#[allow(clippy::identity_op)] // mirrors the shape above\n"
            "fn a() {}\n",
            [],
        ),
    ):
        names = [name for _, name in un_ignored_tests(source)]
        if names != expected:
            raise AssertionError(f"source scan fixture returned {names!r}, expected {expected!r}: {source!r}")

    # Suite-isolation fixtures pin both sides of the structural guarantee. The
    # positive controls make a deleted or delayed acquisition fail, while the
    # negative controls keep comments, strings, and a neighbour's lock from
    # satisfying the check.
    for source, expected in (
        ("#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n    run();\n}\n", []),
        ("#[test]\nfn a() {\n    run();\n}\n", ["a"]),
        (
            "#[test]\nfn a() {\n    construct_ep();\n    let _suite_lock = lock_gpu();\n}\n",
            ["a"],
        ),
        (
            '#[test]\nfn a() {\n    let note = "let _suite_lock = lock_gpu();";\n    run();\n}\n',
            ["a"],
        ),
        (
            "#[test]\nfn a() {\n    // let _suite_lock = lock_gpu();\n    run();\n}\n",
            ["a"],
        ),
        (
            "#[test]\nfn first() {\n    let _suite_lock = lock_gpu();\n}\n"
            "#[test]\nfn second() {\n    run();\n}\n",
            ["second"],
        ),
        (
            "#[test]\nfn a() {\n    let _ = lock_gpu();\n    run();\n}\n",
            ["a"],
        ),
        # Acquiring first is insufficient if the guard is explicitly ended.
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    drop(_suite_lock);\n    run();\n}\n",
            ["a"],
        ),
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    std::mem::drop(_suite_lock);\n    run();\n}\n",
            ["a"],
        ),
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    { core::mem::drop(_suite_lock); }\n    run();\n}\n",
            ["a"],
        ),
        # Rebinding, assignment, and moves can also end the original guard.
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    let _suite_lock = ();\n    run();\n}\n",
            ["a"],
        ),
        (
            "#[test]\nfn a() {\n    let mut suite_lock = lock_gpu();\n"
            "    suite_lock = lock_gpu();\n    run();\n}\n",
            ["a"],
        ),
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    consume(_suite_lock);\n    run();\n}\n",
            ["a"],
        ),
        # Masked comments and literals must not look like guard uses.
        (
            "#[test]\nfn a() {\n    let _suite_lock = lock_gpu();\n"
            "    // std::mem::drop(_suite_lock);\n"
            "    let note = \"drop(_suite_lock)\";\n    run();\n}\n",
            [],
        ),
    ):
        names = [item.name for item in tests_missing_suite_lock(source, "lock_gpu")]
        if names != expected:
            raise AssertionError(
                f"suite-lock scan fixture returned {names!r}, expected {expected!r}: {source!r}"
            )
    helper_fixture = (
        "static GPU_LOCK: Mutex<()> = Mutex::new(());\n"
        "fn lock_gpu() -> MutexGuard<'static, ()> {\n"
        "    GPU_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())\n"
        "}\n"
    )
    helper = function_body(helper_fixture, "lock_gpu")
    if helper is None or not re.search(r"\bGPU_LOCK\s*\.\s*lock\s*\(\s*\)", helper):
        raise AssertionError("suite-lock helper scan cannot observe a known lock acquisition")

    def isolated_target_fixture(lock_function: str, lock_static: str) -> str:
        return (
            f"static {lock_static}: Mutex<()> = Mutex::new(());\n"
            f"fn {lock_function}() -> MutexGuard<'static, ()> {{\n"
            f"    {lock_static}.lock().unwrap_or_else(|poisoned| poisoned.into_inner())\n"
            "}\n"
            "#[test]\n"
            "fn exercises_gpu() {\n"
            f"    let _lock = {lock_function}();\n"
            "    run();\n"
            "}\n"
        )

    healthy_isolation_sources = {
        target: isolated_target_fixture(lock_function, lock_static)
        for target, (lock_function, lock_static) in SERIALIZED_TARGET_LOCKS.items()
    }
    if suite_isolation_errors(SERIALIZED_TARGET_LOCKS, healthy_isolation_sources):
        raise AssertionError("the complete suite-isolation census must pass")

    empty_map_errors = suite_isolation_errors({}, {})
    if len(empty_map_errors) != 1 or "configuration is empty" not in empty_map_errors[0]:
        raise AssertionError("an empty isolation map must fail as an empty configuration")

    removed_target = dict(SERIALIZED_TARGET_LOCKS)
    removed_target.pop("dft_gpu")
    removed_errors = suite_isolation_errors(removed_target, healthy_isolation_sources)
    if not any(
        "target census does not match policy" in error and "dft_gpu" in error
        for error in removed_errors
    ):
        raise AssertionError(
            "removing one configured target must fail the independent policy census"
        )

    zero_test_sources = dict(healthy_isolation_sources)
    zero_test_sources["stft_gpu"] = helper_fixture.replace("GPU_LOCK", "STFT_GPU_LOCK").replace(
        "lock_gpu", "lock_stft_gpu"
    )
    zero_test_errors = suite_isolation_errors(SERIALIZED_TARGET_LOCKS, zero_test_sources)
    if not any(
        "stft_gpu.rs" in error and "contains no #[test] items" in error
        for error in zero_test_errors
    ):
        raise AssertionError(
            "a mapped source with zero tests must fail instead of passing vacuously"
        )

    missing_acquisition_sources = dict(healthy_isolation_sources)
    missing_acquisition_sources["multi_head_attention_gpu"] = missing_acquisition_sources[
        "multi_head_attention_gpu"
    ].replace("    let _lock = lock_mha_gpu();\n", "")
    missing_acquisition_errors = suite_isolation_errors(
        SERIALIZED_TARGET_LOCKS, missing_acquisition_sources
    )
    if not any(
        "multi_head_attention_gpu.rs" in error
        and "exercises_gpu must acquire lock_mha_gpu()" in error
        for error in missing_acquisition_errors
    ):
        raise AssertionError("removing one test acquisition must name the unisolated test")

    wrong_lock_map = dict(SERIALIZED_TARGET_LOCKS)
    wrong_lock_map["nms_gpu"] = ("lock_nms_gpu", "NOT_THE_NMS_LOCK")
    wrong_lock_errors = suite_isolation_errors(wrong_lock_map, healthy_isolation_sources)
    if not any(
        "nms_gpu.rs" in error and "NOT_THE_NMS_LOCK" in error for error in wrong_lock_errors
    ):
        raise AssertionError("a wrong configured lock name must fail the helper/static check")

    if mask_source('let s = "a // b"; // c\n')[0][0].rstrip() != 'let s = "______";':
        raise AssertionError("masking must blank literals and comments while preserving columns")

    # The item boundary is a backstop for bracket accounting, so it has to be
    # exercised against accounting that is already wrong -- the stray `)` here
    # stands in for a masking failure. Without the boundary the walk climbs out
    # of `a` and adopts its `#[ignore]`, which is the fail-open this must not
    # have. Driven through `head_line_indices` because no valid Rust reaches
    # this state once masking is correct.
    corrupted = [("#[ignore]", False), ("#[test]", False), ("fn a() {}", False), (")", False), ("#[test]", False)]
    if head_line_indices(corrupted, 4) != [3]:
        raise AssertionError("the upward walk must stop at an item boundary even when brackets do not balance")

    # Inventory-drift fixtures. The rule these encode is that a policed target
    # must offer the *same* tests with `gpu-tests` off as on -- ignored, but
    # present. Each spelling that has actually reached main gets a positive
    # control, and each is paired with the sanctioned spelling as a negative
    # control, because a drift check that flagged `cfg_attr` too would condemn
    # every correctly written test in the tree.
    target_gated = '#![cfg(feature = "gpu-tests")]\n\n#[test]\nfn a() {}\n'
    if target_level_feature_gate(target_gated) != 1:
        raise AssertionError("a target-level gpu-tests gate must be reported with its line")
    # Only `gpu-tests` separates the two configurations, so a cfg on any other
    # feature resolves identically in both and causes no drift. Flagging one
    # would fail the fast lane on code the authoritative check accepts, and
    # `#![cfg(feature = "cuda")]` already exists in this tree.
    for benign in (
        "#![allow(clippy::too_many_lines)]\n",
        '#![cfg(target_os = "linux")]\n',
        '#![cfg(feature = "cuda")]\n',
        '#![cfg(feature = "cuda-13000")]\n',
    ):
        if target_level_feature_gate(benign) is not None:
            raise AssertionError(f"inner attribute {benign!r} causes no inventory drift and must not be flagged")
    # The gate is an inner attribute; the outer `#[cfg_attr]` spelling on a test
    # must not be mistaken for one, or the fix for this class reports itself.
    correct = '#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]\n#[test]\nfn a() {}\n'
    if target_level_feature_gate(correct) is not None:
        raise AssertionError("an outer cfg_attr on a test is not a target-level gate")
    # Prose and string literals must not be able to fabricate a gate, which is
    # the failure the whole masking layer exists to prevent.
    quoted = '//! Formerly #![cfg(feature = "gpu-tests")], now a shim.\nlet s = "#![cfg(feature = \\"gpu-tests\\")]";\n'
    if target_level_feature_gate(quoted) is not None:
        raise AssertionError("a gate named in a comment or a literal is not a gate")
    # Spans must be found in masked text, not merely *read* from it. An
    # unbalanced `#![cfg(` inside a literal makes the bracket walk run to the
    # end of the file without closing, and `attribute_spans` then abandons the
    # rest of the file -- so a real gate below it is never examined at all.
    # Masking blanks the literal, so the walk never starts there.
    shadowed = 'const S: &str = "#![cfg(";\n#![cfg(feature = "gpu-tests")]\n'
    if target_level_feature_gate(shadowed) != 2:
        raise AssertionError("an unbalanced bracket inside a literal must not hide the gate below it")

    per_test_gated = '#[cfg(feature = "gpu-tests")]\n#[test]\nfn a() {}\n'
    gated_items = test_items(per_test_gated)
    if len(gated_items) != 1:
        raise AssertionError("the feature-gated test fixture must yield exactly one item")
    if not feature_gated_attribute(gated_items[0].masked_head, gated_items[0].raw_head):
        raise AssertionError("a cfg on gpu-tests deletes the test from one inventory and must be flagged")
    # A cfg that removes the test from the *other* configuration is drift too.
    inverted = test_items('#[cfg(not(feature = "gpu-tests"))]\n#[test]\nfn a() {}\n')
    if not feature_gated_attribute(inverted[0].masked_head, inverted[0].raw_head):
        raise AssertionError("a negated gpu-tests cfg drifts the other way and must be flagged")
    # `cfg_attr` is the sanctioned spelling and shares the `cfg` prefix; a
    # substring test for "cfg(" would flag it and make the check unusable.
    correct_items = test_items(correct)
    if feature_gated_attribute(correct_items[0].masked_head, correct_items[0].raw_head):
        raise AssertionError("cfg_attr keeps the test in both inventories and must not be flagged")
    # Neither a platform cfg nor a cfg on a feature held constant across both
    # builds moves a test between inventories.
    for harmless in ('#[cfg(target_os = "linux")]', '#[cfg(feature = "cuda")]', '#[cfg(feature = "tracing")]'):
        items = test_items(f"{harmless}\n#[test]\nfn a() {{}}\n")
        if feature_gated_attribute(items[0].masked_head, items[0].raw_head):
            raise AssertionError(f"{harmless} causes no feature-inventory drift and must not be flagged")

    # The feature name is read from raw text, where masking has not blanked it,
    # so the *presence* of a feature key must be established on masked text.
    # Otherwise a feature named inside a string literal is read as a real one.
    literal_feature = test_items('#[cfg(all(unix, target_env = r#"feature = "gpu-tests""#))]\n#[test]\nfn a() {}\n')
    if feature_gated_attribute(literal_feature[0].masked_head, literal_feature[0].raw_head):
        raise AssertionError("a feature named inside a string literal is not a gate")

    # A gate on the module around a test deletes it just as surely, and the
    # head walk stops at the `mod` boundary so it is invisible from inside.
    gated_mod = '#[cfg(feature = "gpu-tests")]\nmod inner {\n    #[test]\n    fn a() {}\n}\n'
    if gpu_tests_gated_modules(gated_mod) != [1]:
        raise AssertionError("a gpu-tests-gated module containing tests must be flagged")
    # Gating a module of *helpers* is the sanctioned two-arm shim, and the
    # two-arm shim is the fix this check recommends -- flagging it would make
    # the advice self-defeating.
    helper_mod = '#[cfg(feature = "gpu-tests")]\nmod shim {\n    pub fn helper() {}\n}\n'
    if gpu_tests_gated_modules(helper_mod):
        raise AssertionError("a gated helper module holds no tests and must not be flagged")
    # The brace walk must end at the module it started in, or the next
    # module's tests are attributed to this one.
    neighbour = '#[cfg(feature = "gpu-tests")]\nmod shim {\n    pub fn helper() {}\n}\n\nmod real {\n    #[test]\n    fn a() {}\n}\n'
    if gpu_tests_gated_modules(neighbour):
        raise AssertionError("a later module's tests must not be attributed to an earlier gated one")
    if gpu_tests_gated_modules('mod inner {\n    #[test]\n    fn a() {}\n}\n'):
        raise AssertionError("an ungated module is not drift")
    # The brace walk must start at the gated module's own body. Starting
    # anywhere earlier lets an unrelated module's tests be counted as this
    # one's, which flags a correct gated helper and reds a green lane.
    preceded = (
        'mod first {\n    #[test]\n    fn a() {}\n}\n\n'
        '#[cfg(feature = "gpu-tests")]\nmod second {\n    pub fn helper() {}\n}\n'
    )
    if gpu_tests_gated_modules(preceded):
        raise AssertionError("an earlier module's tests must not be counted as the gated module's")

    # `required-features` keeps the target from being built at all. Parsed with
    # tomllib because the two shapes below are exactly what a hand-rolled
    # section split gets wrong: a following table absorbed into the preceding
    # test's span, and a header carrying a legal trailing comment.
    if list(manifest_required_features('[[test]]\nname = "foo_gpu"\nrequired-features = ["gpu-tests"]\n')) != ["foo_gpu"]:
        raise AssertionError("required-features on a policed target must be reported")
    if list(manifest_required_features('[[test]] # a comment\nname = "foo_gpu"\nrequired-features = ["gpu-tests"]\n')) != ["foo_gpu"]:
        raise AssertionError("a trailing comment on the table header must not hide required-features")
    absorbed = '[[test]]\nname = "foo_gpu"\n\n[[bench]]\nname = "b"\nrequired-features = ["x"]\n'
    if list(manifest_required_features(absorbed)):
        raise AssertionError("a following bench table must not be read as part of the test target")
    if list(manifest_required_features('[[test]]\nname = "plain_cpu"\nrequired-features = ["gpu-tests"]\n')):
        raise AssertionError("an unpoliced target is not this check's business")
    if list(manifest_required_features('[[test]]\nname = "foo_gpu"\n')):
        raise AssertionError("a policed target without required-features is correct")

    good_active = ActiveResult("fixture_active", inventory=2, passed=0, failed=2, ignored=0)
    silent_with_feature = ActiveResult("fixture_active_silent", inventory=1, passed=1, failed=0, ignored=0)
    if validate_active_no_cuda_result(good_active):
        raise AssertionError("active fail-loud fixture should pass")
    if not any("passed with gpu-tests" in error for error in validate_active_no_cuda_result(silent_with_feature)):
        raise AssertionError("gpu-tests-enabled silent pass should fail")

    # Census fixtures. The two scans above are per-target and report nothing
    # when there are no targets, so `--source-scan` used to exit 0 with the
    # CUDA tests directory deleted -- reporting success for the inventory
    # disappearance it is partly there to catch.
    #
    # These are pure: they pass literal mappings rather than reading the tree,
    # so `--self-test` stays hermetic and the partial case below is expressible
    # at all. It is not otherwise, since it needs one directory populated and
    # one empty at the same instant.
    ep_dir, memory_dir = TEST_DIRS
    healthy = {ep_dir: [ep_dir / "matmul_gpu.rs"], memory_dir: [memory_dir / "pool_gpu.rs"]}
    if census_errors(healthy):
        raise AssertionError("a populated census is the healthy case and must not be flagged")
    if len(census_errors({ep_dir: [], memory_dir: []})) != 2:
        raise AssertionError("a wholly empty census must fail once per policed directory")
    # The case an aggregate non-empty check cannot see: one directory emptied
    # while the other still holds enough targets to look healthy. This is the
    # realistic shape -- these crates have split once already.
    partial = census_errors({ep_dir: [ep_dir / "matmul_gpu.rs"], memory_dir: []})
    if len(partial) != 1 or str(memory_dir.relative_to(ROOT)) not in partial[0]:
        raise AssertionError(
            "one emptied directory must fail and must name itself, even while the other is full"
        )


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
        by_dir = policed_cuda_targets_by_dir()
        policed = [path for targets in by_dir.values() for path in targets]
        source_errors = (
            scan_source_for_missing_ignores()
            + scan_source_for_inventory_drift()
            + scan_source_for_suite_isolation()
            + census_errors(by_dir)
        )
        if source_errors:
            print("CUDA test source scan failed:", file=sys.stderr)
            for error in source_errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print(
            f"CUDA test source scan passed over {len(policed)} CUDA test targets: "
            "every test carries an ignore and is present with gpu-tests off; "
            f"{len(EXPECTED_SUITE_ISOLATION_TARGETS)} resource-sensitive targets acquire their "
            "suite lock before test work and keep its guard live through function exit"
        )
        return 0

    # The census runs here too, not just under --source-scan. The inventory
    # guard below keys on what Cargo reported; this one keys on what this
    # script's own TEST_DIRS found. A crate layout that moved out from under
    # TEST_DIRS leaves Cargo reporting tests normally while both source scans
    # silently examine nothing, and only this catches that.
    errors: list[str] = (
        scan_source_for_missing_ignores()
        + scan_source_for_inventory_drift()
        + scan_source_for_suite_isolation()
        + census_errors(policed_cuda_targets_by_dir())
    )
    for crate in CUDA_CRATES:
        manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
        if not declares_gpu_tests_feature(manifest):
            errors.append(f"crates/{crate.name}/Cargo.toml must define a gpu-tests feature")

    base_binaries = build_test_binaries(BASE_CONFIG)
    gpu_binaries = build_test_binaries(GPU_CONFIG)
    base_inventory = collect_inventories(base_binaries)
    gpu_inventory = collect_inventories(gpu_binaries)
    errors.extend(compare_inventories(base_inventory, gpu_inventory))
    production_summaries, production_errors = verify_production_feature_targets()
    errors.extend(production_errors)

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
    for summary in production_summaries:
        print(f"Production feature target census passed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
