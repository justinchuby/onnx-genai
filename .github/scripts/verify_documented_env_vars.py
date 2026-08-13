#!/usr/bin/env python3
"""Fail when the docs advertise an environment variable no code reads.

# Why this exists

A user who sets a documented environment variable that nothing parses gets **no
error and no effect**. That is indistinguishable from the variable working and
having no visible consequence, so the mistake survives indefinitely.

This is the same shape as the failure recorded in `docs/MEMORY_ARCHITECTURE.md`
under "How this area fails" -- not wrong code, but code (or in this case
configuration) that is documented, plausible, and unreachable. `device_policy` /
`gpu_layers:N` was documented for months with no parser anywhere in the
workspace (#678), and `ONNX_GENAI_WEIGHT_PREFETCH` sat in a block of active
environment aliases with zero code references.

# The rule

Every `ONNX_GENAI_*` or `NXRT_*` name appearing in `docs/` must either

* appear somewhere in `crates/**/*.rs`, or
* appear in a repository script (`scripts/**/*.{py,sh,ps1}`), or
* be listed in `KNOWN_UNIMPLEMENTED` below, with a reason.

Scripts count because a documented knob a repository runner reads is
implemented, and the failure this gate exists to catch — a name a user can set
with no error and no effect — is the same whether the reader is a crate or a
runner. The allowlist is unaffected: no `KNOWN_UNIMPLEMENTED` entry appears in
`scripts/`, so widening the search cannot turn an honest caveat into a false
"implemented".

Adding a variable to the allowlist is deliberate and reviewable. Forgetting to
wire one up is not silent.

# Filename-reference exclusion

A match immediately followed by a documentation file extension (`.md`, `.rst`,
`.toml`) is a **filename**, not an environment variable — e.g. `NXRT_ABI.md` is
a cross-reference to `docs/NXRT_ABI.md`, not an undeclared knob. A genuine env
var is never written with a file extension suffix, so this exclusion cannot mask
a real gap. Without this rule, any doc named with the `NXRT_*` or
`ONNX_GENAI_*` prefix would trip the gate the moment another doc links to it —
a false-positive class, not a one-off.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Names the docs mention specifically to say they are *not* implemented. Each
# entry must stay accompanied by that statement in the prose; the point of
# listing them here is that dropping the caveat and leaving the name is exactly
# the mistake this script exists to catch, so the two must be changed together.
KNOWN_UNIMPLEMENTED: dict[str, str] = {
    "ONNX_GENAI_WEIGHT_BUDGET": "documented in WEIGHT_OFFLOAD.md as not implemented",
    "ONNX_GENAI_WEIGHT_DEVICE_BUDGET": "documented in WEIGHT_OFFLOAD.md as not implemented",
    "ONNX_GENAI_WEIGHT_HOST_BUDGET": "documented in WEIGHT_OFFLOAD.md as not implemented",
    "ONNX_GENAI_GPU_LAYERS": "documented in WEIGHT_OFFLOAD.md as not implemented; "
    "use serving.memory.weights.device_policy = gpu_layers:N instead (#678)",
    "ONNX_GENAI_WEIGHT_PREFETCH": "documented in WEIGHT_OFFLOAD.md as not implemented",
    "NXRT_AUTO_INSTALL_CUDA": "referenced in prose about a proposed installer step",
    "NXRT_SQNBIT_PREFILL_MIN": "referenced in prose about a proposed tuning knob",
    "ONNX_GENAI_BASE_URL": "client-side variable read by external tooling, not by this workspace",
    "ONNX_GENAI_SD_PACKAGE": "referenced in prose about a proposed packaging layout",
}

# Uppercase only: the lowercase `onnx_genai_*` names in the docs are Prometheus
# metric identifiers, which are a different surface with a different contract.
# The negative lookahead excludes filename references (see docstring above).
ENV_PATTERN = re.compile(r"\b(?:ONNX_GENAI|NXRT)_[A-Z0-9_]+\b(?!\.(?:md|rst|toml)\b)")


def collect(root: Path, pattern: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in root.rglob(pattern):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in ENV_PATTERN.findall(text):
            found.setdefault(name, set()).add(str(path))
    return found


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    documented = collect(repo / "docs", "*.md")
    in_code = collect(repo / "crates", "*.rs")
    for pattern in ("*.py", "*.sh", "*.ps1"):
        for name, where in collect(repo / "scripts", pattern).items():
            in_code.setdefault(name, set()).update(where)

    missing = {
        name: sorted(where)
        for name, where in documented.items()
        if name not in in_code and name not in KNOWN_UNIMPLEMENTED
    }

    # The reverse check: an allowlisted name that has since been implemented
    # should leave the allowlist, or the list rots into a permanent excuse.
    stale = sorted(name for name in KNOWN_UNIMPLEMENTED if name in in_code)

    if not missing and not stale:
        print(
            f"env var honesty: {len(documented)} documented, "
            f"{len(KNOWN_UNIMPLEMENTED)} known-unimplemented, all accounted for"
        )
        return 0

    print("Documented environment variable honesty check failed:", file=sys.stderr)
    for name, where in sorted(missing.items()):
        files = ", ".join(Path(p).name for p in where)
        print(
            f"  - {name}: documented in {files} but no crate reads it. "
            f"Wire it up, or add it to KNOWN_UNIMPLEMENTED with a reason and say "
            f"so in the prose.",
            file=sys.stderr,
        )
    for name in stale:
        print(
            f"  - {name}: listed as unimplemented but a crate now reads it. "
            f"Remove it from KNOWN_UNIMPLEMENTED and drop the caveat from the docs.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
