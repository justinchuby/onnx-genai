#!/usr/bin/env python3
"""Compare criterion benchmark results between a PR and its merge-base.

This is a REGRESSION GATE, not an advisory comment. It fails (exit 1) when a
benchmark scenario exceeds the failure threshold, blocking the PR until the
regression is addressed or justified.

Methodology
-----------
The workflow benchmarks the merge-base FIRST (absorbs cold-start/cache-miss
overhead), then the PR head SECOND (benefits from warm caches and steady-state
runner). This creates a small systematic bias toward PR appearing faster, which:
  - Reduces false positives (PR must overcome warm-runner advantage to appear
    slower, so only genuine regressions trip the gate).
  - Makes the gate more conservative and therefore more durable — a gate that
    cries wolf gets disabled within a week.

The comparison uses criterion point-estimate means from 100 samples each.

Threshold derivation
--------------------
Measured variance on the CI runner (macOS arm64 M1 Virtual, 3 cores) with the
CORRECTED ordering (base-first, PR-second):
  - Single-threaded kernels: typically <10% noise
  - Multi-threaded matmul: up to ~20% noise (fewer cores = more contention)
  - Hot-path (sampling, tokenization): <5% noise

The failure threshold (30%) is set at 1.5× the worst observed single-scenario
noise (~20%), giving a margin that prevents false positives while catching the
historical regression class we guard against (350% = 4.5× M=1 decode).

Signal-to-noise margin: 350% / 30% = 11.7×.

If CI noise proves larger than expected after deployment, raise the threshold
from the data — but document exactly what variance was observed on what
scenarios, so the next person can re-derive rather than guess.

Exit codes
----------
0 — all changes within threshold (PR may merge)
1 — at least one regression exceeds the failure threshold (PR BLOCKED)

Known gaps
----------
This script compares criterion point estimates. It cannot detect:
  - Regressions in code paths not exercised by the benchmark suite.
  - Sub-threshold regressions that compound over multiple PRs.
  - Regressions that only manifest with a real model (GQA fusion, speculative
    decode, end-to-end token throughput).
  - GPU-only regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Thresholds ───────────────────────────────────────────────────────────────
#
# FAIL_THRESHOLD: the gate trips here. Derived from:
#   - Observed CI noise: ~20% worst-case on multi-threaded matmul with
#     base-first ordering on 3-core M1 Virtual runner.
#   - Margin: 1.5× worst noise → 30%.
#   - Historical regression: 4.5× (350%). Signal-to-noise: 11.7×.
#
# WARN_THRESHOLD: surfaces suspicious changes (does NOT block).
#
# To re-derive: run the benchmark twice on the same commit with base-first
# ordering and record the max positive delta across all scenarios. Set
# FAIL_THRESHOLD at 1.5× that value.
WARN_THRESHOLD = 0.15   # 15% slower → ⚠️ in comment
FAIL_THRESHOLD = 0.30   # 30% slower → 🔴 gate FAILS


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
        header = "## 🔴 Benchmark Regression — Check FAILED"
    elif has_warning:
        header = "## ⚠️ Benchmark Change Detected"
    else:
        header = "## ✅ Benchmarks — No Regression"

    lines = [
        header,
        "",
    ]

    if has_regression:
        lines.append(
            "**This check has FAILED.** One or more scenarios regressed beyond "
            f"the {fail:.0%} threshold. The PR is blocked until the regression is "
            "addressed or the threshold is re-evaluated against measured noise."
        )
        lines.append("")

    lines.extend([
        "Comparison of criterion micro-benchmarks: **PR head vs merge-base**, "
        "measured on the same runner in the same job (base first → PR second).",
        "",
        "> ℹ️ Absolute times are **informational only** — they vary with runner "
        "load. The **% change** column is the reliable signal because both sides "
        "ran under identical conditions.",
        "",
        "| Status | Scenario | Base | PR | Change |",
        "|:---:|---|---:|---:|---:|",
    ])

    for scenario, base_ns, pr_ns, change in rows:
        icon = status_icon(change, warn, fail)
        sign = "+" if change >= 0 else ""
        lines.append(
            f"| {icon} | `{scenario}` | {format_ns(base_ns)} "
            f"| {format_ns(pr_ns)} | {sign}{change:.1%} |"
        )

    lines.append("")
    lines.append(
        f"**Thresholds:** ⚠️ ≥ {warn:.0%} slower (advisory), "
        f"🔴 ≥ {fail:.0%} slower (**blocks merge**) — "
        "derived from measured runner variance × 1.5 safety margin"
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
