#!/usr/bin/env python3
"""Check that platform-specific kernel files have platform markers in their names.

A file whose name sounds platform-neutral but whose body is *entirely* gated
behind a single architecture is a latent defect: contributors and reviewers
assume the code covers every target, so a missing fallback goes unnoticed.

This is not hypothetical.  The crate has been bitten multiple times:

  - ``simd_gemm.rs`` and ``bf16_gemm.rs`` had platform-neutral names but were
    100 % x86 code.  No equivalent aarch64 GEMM existed, but the names hid
    that fact from every reviewer who read the directory listing.

  - ``dot_f32`` / ``axpy_f32`` in ``sdpa.rs`` had AVX2 paths and *no* aarch64
    path, so Apple Silicon silently fell to scalar -- nobody noticed for months
    because nothing in the name or location said "x86 only."

The rule
--------
A ``.rs`` file inside the CPU EP kernel directory is **flagged** when ALL of
the following hold:

1. It contains ``cfg`` gates referencing exactly **one** architecture family
   (x86, aarch64, macOS) and **zero** gates for any other family.
2. It has **no unconditionally-compiled top-level items** -- every ``fn``,
   ``const``, ``struct``, ``enum``, ``impl``, ``trait``, ``type``, ``mod``,
   ``use``, ``static``, or ``extern`` at column 0 is preceded by a
   platform-specific ``#[cfg(...)]`` attribute.
3. Its filename stem does **not** already contain a recognized platform marker
   (``x86``, ``neon``, ``accelerate``, etc.).

Condition 2 is the key false-positive filter.  Files like ``simd_normalize.rs``
or ``simd_quant.rs`` have *portable entry points* that dispatch to x86 SIMD
internally and fall back to scalar -- they compile and work on every platform.
Those are NOT flagged: they have unconditionally-compiled public functions.

Known gap
---------
This lint catches files that are *entirely* single-platform but does NOT catch
files with portable entry points that contain single-platform *helpers* inside
them (the ``sdpa.rs`` / ``dot_f32`` case).  That is a different class of bug --
a missing-implementation problem, not a naming problem -- and requires a
different check (e.g. "every SIMD helper has a scalar fallback").  See the
``kernels/gemm/`` layout plan for the broader strategy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "crates" / "onnx-runtime-ep-cpu" / "src" / "kernels"

# Platform markers that, if present in a filename stem, signal that the file is
# intentionally platform-specific.  Case-insensitive match.
PLATFORM_MARKERS = {
    "x86", "avx", "sse",                  # x86 family
    "aarch64", "neon", "sve",              # ARM family
    "accelerate", "bnns", "amx",           # macOS Accelerate framework
    "mlas",                                # vendored MLAS (x86 only today)
    "xnnpack",                             # Android / XNNPACK
    "qnn",                                 # Qualcomm QNN
}

# Architecture families and the cfg patterns that identify them.
ARCH_FAMILIES: dict[str, list[re.Pattern[str]]] = {
    "x86": [
        re.compile(r'cfg\s*\(.*target_arch\s*=\s*"x86(?:_64)?"'),
        re.compile(r"is_x86_feature_detected!"),
        re.compile(r"std::arch::x86(?:_64)?::"),
    ],
    "aarch64": [
        re.compile(r'cfg\s*\(.*target_arch\s*=\s*"aarch64"'),
        re.compile(r"is_aarch64_feature_detected!"),
        re.compile(r"std::arch::aarch64::"),
    ],
    "macos": [
        re.compile(r'cfg\s*\(.*target_os\s*=\s*"macos"'),
        re.compile(r'cfg\s*\(.*target_os\s*=\s*"ios"'),
        re.compile(r'link\s*\(\s*name\s*=\s*"Accelerate"'),
    ],
}

# Matches a top-level Rust item definition at column 0 (no leading whitespace).
ITEM_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:async\s+)?(?:const\s+)?"
    r"(?:fn|struct|enum|trait|type|mod|use|impl|static|extern)\b"
)

# Matches a #[cfg(...)] or #[cfg_attr(...)] that references a specific target
# architecture or OS.
PLATFORM_CFG_RE = re.compile(
    r"#\[cfg(?:_attr)?\s*\(.*(?:target_arch|target_os)\s*="
)


def families_in_file(path: Path) -> set[str]:
    """Return the set of architecture families whose cfg patterns appear."""
    text = path.read_text()
    found: set[str] = set()
    for family, patterns in ARCH_FAMILIES.items():
        if any(p.search(text) for p in patterns):
            found.add(family)
    return found


def has_portable_items(path: Path) -> bool:
    """True if the file has at least one top-level item not preceded by a
    platform-specific ``#[cfg]`` attribute.

    A "portable item" is a ``fn``, ``struct``, ``const``, etc. at column 0
    whose immediately-preceding attribute block does NOT contain a
    ``target_arch`` or ``target_os`` cfg gate.  Comments, doc-comments, blank
    lines, and non-platform attributes (``#[inline]``, ``#[derive(...)]``,
    ``#[allow(...)]``) are skipped during the lookback.
    """
    lines = path.read_text().splitlines()

    for i, line in enumerate(lines):
        if not ITEM_RE.match(line):
            continue

        # Look backward through the attribute block for a platform cfg.
        gated = False
        for j in range(i - 1, max(i - 20, -1), -1):
            prev = lines[j].strip()
            if not prev or prev.startswith("//"):
                continue
            if PLATFORM_CFG_RE.search(prev):
                gated = True
                break
            if prev.startswith("#["):
                # Non-platform attribute — keep looking.
                continue
            # Non-attribute, non-comment, non-blank line: stop lookback.
            break

        if not gated:
            return True

    return False


def has_platform_marker(stem: str) -> bool:
    """Whether the filename stem contains any recognized platform marker."""
    lower = stem.lower()
    return any(marker in lower for marker in PLATFORM_MARKERS)


def main() -> int:
    if not KERNELS.is_dir():
        sys.exit(f"kernel directory not found: {KERNELS}")

    problems: list[str] = []
    scanned = 0

    for rs_file in sorted(KERNELS.rglob("*.rs")):
        scanned += 1
        families = families_in_file(rs_file)

        # No arch-specific code, or multi-arch code: fine either way.
        if len(families) != 1:
            continue

        # Filename already carries a platform marker: fine.
        if has_platform_marker(rs_file.stem):
            continue

        # File has unconditionally-compiled top-level items: it is a portable
        # file with platform-specific *optimization branches*, not a
        # platform-specific file.  This is the key false-positive filter.
        if has_portable_items(rs_file):
            continue

        (family,) = families
        rel = rs_file.relative_to(REPO)
        problems.append(
            f"  {rel}\n"
            f"    Every top-level item is gated behind {family!r} cfg, but the\n"
            f"    filename has no platform marker — this hides the fact that the\n"
            f"    file compiles to nothing on other platforms.\n"
            f"\n"
            f"    History: simd_gemm.rs and bf16_gemm.rs had platform-neutral\n"
            f"    names but were 100% x86 code; dot_f32/axpy_f32 in sdpa.rs had\n"
            f"    AVX2 paths and no aarch64 path — Apple Silicon silently fell to\n"
            f"    scalar for months because nothing in the name said 'x86 only.'\n"
            f"\n"
            f"    Fix: rename to include a marker (e.g. x86_, neon_, accelerate_)\n"
            f"    or add a portable fallback so the file is truly cross-platform."
        )

    if problems:
        print(
            "platform-naming lint: files with single-arch code must carry a "
            "platform marker in their filename:\n"
        )
        for p in problems:
            print(p)
        print(
            f"\nRecognized markers (case-insensitive, in filename stem): "
            f"{', '.join(sorted(PLATFORM_MARKERS))}"
        )
        return 1

    print(
        f"platform-naming lint: {scanned} kernel file(s) checked, "
        f"no unmarked single-arch files found"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
