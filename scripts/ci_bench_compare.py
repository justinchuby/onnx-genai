#!/usr/bin/env python3
"""Compare criterion benchmark results between a PR and its merge-base.

Reads criterion JSON estimates from two directories (base and PR), computes
relative change for each scenario, and emits a Markdown table suitable for
posting as a PR comment. Reports percentage change (the ratio that matters
for regression detection) and treats absolute timings as informational only.

Exit codes
----------
0 — all changes within threshold
1 — at least one regression exceeds the failure threshold

Known gaps
----------
This script compares criterion point estimates. It cannot detect:
  - Regressions in code paths not exercised by the benchmark suite.
  - Noise-driven false positives when CI runners have extreme load variance
    (mitigated by running base and PR on the same runner in the same job).
  - Sub-threshold regressions that compound over multiple PRs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# Default thresholds, calibrated from measured run-to-run variance.
#
# On a macOS arm64 developer machine, two identical-code benchmark runs
# showed up to ~48% variance on multi-threaded matmul (due to background
# load) and ~15% on single-threaded kernels. CI runners (dedicated VMs)
# show lower variance but are not zero-noise.
#
# The regression this workflow exists to catch was 4.5× (350%). With a
# 50% failure threshold, we have ≥7× signal-to-noise margin for that
# class of regression, while avoiding false positives from normal jitter.
#
# A 15% warning threshold surfaces suspicious changes for human review
# without blocking the build.
WARN_THRESHOLD = 0.15   # 15% slower → warning
FAIL_THRESHOLD = 0.50   # 50% slower → failure


def parse_estimates(criterion_dir: Path) -> dict[str, float]:
    """Walk a criterion output tree and return {scenario: mean_ns}."""
    results: dict[str, float] = {}
    for estimate in criterion_dir.glob("**/new/estimates.json"):
        relative = estimate.relative_to(criterion_dir)
        scenario = "/".join(relative.parts[:-2])
        if scenario.startswith(("report/", "change/")):
            continue
        data = json.loads(estimate.read_text())
        results[scenario] = data["mean"]["point_estimate"]
    return results


def compute_changes(
    base: dict[str, float], pr: dict[str, float]
) -> list[tuple[str, float, float, float]]:
    """Return [(scenario, base_ns, pr_ns, change_ratio)] sorted by change."""
    rows = []
    for scenario in sorted(set(base) | set(pr)):
        base_ns = base.get(scenario)
        pr_ns = pr.get(scenario)
        if base_ns is None or pr_ns is None:
            continue
        if base_ns == 0:
            continue
        change = (pr_ns - base_ns) / base_ns
        rows.append((scenario, base_ns, pr_ns, change))
    rows.sort(key=lambda r: -r[3])
    return rows


def format_ns(ns: float) -> str:
    """Human-friendly time formatting."""
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} µs"
    return f"{ns:.1f} ns"


def status_icon(change: float, warn: float, fail: float) -> str:
    """Return emoji status for a change ratio."""
    if change >= fail:
        return "🔴"
    if change >= warn:
        return "⚠️"
    if change <= -warn:
        return "🟢"
    return "✅"


def generate_markdown(
    rows: list[tuple[str, float, float, float]],
    warn: float,
    fail: float,
    host_info: str = "",
) -> str:
    """Generate the PR comment body."""
    has_regression = any(change >= fail for _, _, _, change in rows)
    has_warning = any(change >= warn for _, _, _, change in rows)

    if has_regression:
        header = "## 🔴 Benchmark Regression Detected"
    elif has_warning:
        header = "## ⚠️ Benchmark Change Detected"
    else:
        header = "## ✅ Benchmarks — No Regression"

    lines = [
        header,
        "",
        "Comparison of criterion micro-benchmarks: **PR head vs merge-base**, "
        "measured on the same runner in the same job.",
        "",
        "> ℹ️ Absolute times are **informational only** — they vary with runner "
        "load. The **% change** column is the reliable signal because both sides "
        "ran under identical conditions.",
        "",
        "| Status | Scenario | Base | PR | Change |",
        "|:---:|---|---:|---:|---:|",
    ]

    for scenario, base_ns, pr_ns, change in rows:
        icon = status_icon(change, warn, fail)
        sign = "+" if change >= 0 else ""
        lines.append(
            f"| {icon} | `{scenario}` | {format_ns(base_ns)} "
            f"| {format_ns(pr_ns)} | {sign}{change:.1%} |"
        )

    lines.append("")
    lines.append(
        f"**Thresholds:** ⚠️ ≥ {warn:.0%} slower, "
        f"🔴 ≥ {fail:.0%} slower (derived from measured runner variance)"
    )

    if host_info:
        lines.append(f"\n<details><summary>Host info</summary>\n\n```\n{host_info}\n```\n</details>")

    lines.append(
        "\n<details><summary>What this cannot catch</summary>\n\n"
        "- Regressions in code paths not covered by these benchmarks "
        "(e.g., end-to-end decode with a real model)\n"
        "- Sub-threshold regressions that compound over multiple PRs\n"
        "- Performance changes that only manifest under GPU execution\n"
        "- Latency changes in the ORT integration path "
        "(these benchmarks exercise the native Rust kernels)\n"
        "</details>"
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Path to criterion results for the merge-base",
    )
    parser.add_argument(
        "--pr-dir",
        type=Path,
        required=True,
        help="Path to criterion results for the PR head",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write markdown to this file (default: stdout)",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=WARN_THRESHOLD,
        help=f"Warn threshold as fraction (default: {WARN_THRESHOLD})",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=FAIL_THRESHOLD,
        help=f"Fail threshold as fraction (default: {FAIL_THRESHOLD})",
    )
    parser.add_argument(
        "--host-info",
        type=str,
        default="",
        help="Optional host information string for the comment",
    )
    args = parser.parse_args()

    if not args.base_dir.is_dir():
        print(f"error: base directory not found: {args.base_dir}", file=sys.stderr)
        return 1
    if not args.pr_dir.is_dir():
        print(f"error: PR directory not found: {args.pr_dir}", file=sys.stderr)
        return 1

    base = parse_estimates(args.base_dir)
    pr = parse_estimates(args.pr_dir)

    if not base:
        print("error: no benchmark results found in base directory", file=sys.stderr)
        return 1
    if not pr:
        print("error: no benchmark results found in PR directory", file=sys.stderr)
        return 1

    rows = compute_changes(base, pr)
    if not rows:
        print("error: no common scenarios between base and PR", file=sys.stderr)
        return 1

    markdown = generate_markdown(
        rows, args.warn_threshold, args.fail_threshold, args.host_info
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"wrote {args.output}")
    else:
        print(markdown)

    # Exit 1 only on unambiguous regression
    has_failure = any(change >= args.fail_threshold for _, _, _, change in rows)
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
