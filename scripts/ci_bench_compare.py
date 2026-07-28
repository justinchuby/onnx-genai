#!/usr/bin/env python3
"""Compare criterion benchmark results between a PR and its merge-base.

Generates a Markdown table for posting as a PR comment. This is an
INFORMATIONAL comparison — it does not block CI. The real regression gates
live in crates/onnx-genai-bench/tests/profile_native.rs (throughput floors
and dispatch-reachability tests on real hardware).

This workflow's value is visibility: surface the comparison on the PR that
caused a change, while the author is still looking at it. A merge earlier
today cost 60.41 → 13.37 tok/s and nobody noticed for hours; a comment on
that PR would have been enough.

Methodology
-----------
The workflow benchmarks the merge-base FIRST (absorbs cold-start overhead),
then the PR head SECOND (warm caches, steady-state runner). This means PR
numbers tend to appear slightly faster — a delta that appears despite this
bias is more likely genuine.

Measured run-to-run variance (same code, base-first ordering, macOS arm64
M1 Virtual 3-core CI runner):
  - Single-threaded kernels: typically <10%
  - Multi-threaded matmul: up to ~27% (3 cores = high contention)
  - Hot-path (sampling, tokenization): <5%

The ⚠️ icon at 15% and 🔴 at 30% are VISUAL FLAGS for the reviewer, not
build gates. A reader needs to know whether a 5% delta is signal or noise,
and these icons provide that context calibrated against measured variance.

Exit codes
----------
Always 0 — this script never fails the build.

Known gaps
----------
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


# ── Visual thresholds ────────────────────────────────────────────────────────
#
# These are ICONS for the reviewer, not build gates.
#
# Calibrated against measured CI runner noise (macOS arm64 M1 Virtual):
#   - Worst observed same-code delta: ~27% (multi-threaded matmul)
#   - WARN at 15%: flags scenarios worth a second look
#   - ALERT at 30%: flags scenarios very likely to be real regressions
#     (above worst observed noise)
#
# The historical regression this exists to surface: 4.5× (350%) on M=1
# decode. At 350%, even the most conservative reader sees the 🔴 immediately.
WARN_THRESHOLD = 0.15   # 15% slower → ⚠️
ALERT_THRESHOLD = 0.30  # 30% slower → 🔴


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


def status_icon(change: float, warn: float, alert: float) -> str:
    """Return emoji status for a change ratio."""
    if change >= alert:
        return "🔴"
    if change >= warn:
        return "⚠️"
    if change <= -warn:
        return "🟢"
    return "✅"


def generate_markdown(
    rows: list[tuple[str, float, float, float]],
    warn: float,
    alert: float,
    host_info: str = "",
) -> str:
    """Generate the PR comment body."""
    has_alert = any(change >= alert for _, _, _, change in rows)
    has_warning = any(change >= warn for _, _, _, change in rows)

    if has_alert:
        header = "## 🔴 Benchmark Regression Detected"
    elif has_warning:
        header = "## ⚠️ Benchmark Change Detected"
    else:
        header = "## ✅ Benchmarks — No Regression"

    lines = [
        header,
        "",
        "Comparison of criterion micro-benchmarks: **PR head vs merge-base**, "
        "measured on the same runner in the same job (base first → PR second).",
        "",
        "> ℹ️ Absolute times are **informational only** — they vary with runner "
        "load. The **% change** column is the reliable signal because both sides "
        "ran under identical conditions.",
        "",
        "| Status | Scenario | Base | PR | Change |",
        "|:---:|---|---:|---:|---:|",
    ]

    for scenario, base_ns, pr_ns, change in rows:
        icon = status_icon(change, warn, alert)
        sign = "+" if change >= 0 else ""
        lines.append(
            f"| {icon} | `{scenario}` | {format_ns(base_ns)} "
            f"| {format_ns(pr_ns)} | {sign}{change:.1%} |"
        )

    lines.append("")
    lines.append(
        f"**Visual flags:** ⚠️ ≥ {warn:.0%} slower, "
        f"🔴 ≥ {alert:.0%} slower — calibrated against "
        "measured runner noise (~27% worst-case on multi-threaded matmul)"
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
    parser = argparse.ArgumentParser(
        description="Compare criterion benchmark results and generate a PR comment."
    )
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

    markdown = generate_markdown(rows, WARN_THRESHOLD, ALERT_THRESHOLD, args.host_info)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"wrote {args.output}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
