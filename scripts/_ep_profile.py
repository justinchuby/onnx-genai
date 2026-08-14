"""Shared helper: count executed nodes per execution provider from an ORT profile.

The ONNX Runtime profiler (`SessionOptions.enable_profiling`) writes a Chrome
trace JSON. Every executed kernel emits an event whose `cat` is `"Node"`, whose
`name` ends in `_kernel_time`, and whose `args` dict carries a `"provider"`
field naming the EP that ran the node (e.g. `"CPUExecutionProvider"` or, for a
plugin EP, the registered EP name such as `"cuda_ep"`).

Counting these events is the *only* trustworthy signal that a plugin EP actually
did work: a session reporting `cuda_ep` in `get_providers()` merely means the EP
was registered, not that it claimed or executed a single node. A validation that
skips this check is consistent with total silent CPU fallback (issue #956).
"""

from __future__ import annotations

import json
from collections import Counter


def count_nodes_by_provider(profile_path: str) -> Counter:
    """Return a Counter mapping provider name -> number of executed nodes.

    Reads the ORT profile JSON at `profile_path` and tallies one count per
    node kernel-time event, keyed by its `args.provider` field.
    """
    with open(profile_path, "r", encoding="utf-8") as fh:
        events = json.load(fh)

    counts: Counter = Counter()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "Node":
            continue
        name = ev.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        provider = ev.get("args", {}).get("provider")
        if provider:
            counts[provider] += 1
    return counts


def format_counts(counts: Counter) -> str:
    if not counts:
        return "  (no node kernel_time events found in profile)"
    total = sum(counts.values())
    lines = [
        f"  {prov}: {n}"
        for prov, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    lines.append(f"  TOTAL executed nodes: {total}")
    return "\n".join(lines)
