#!/usr/bin/env python3
"""Check that performance-gated cfg blocks have fallback coverage instrumentation.

The rule
--------
Any `#[cfg(feature = "mlas")]` block (or similar unreachable-on-target gate)
that guards a performance fast path inside a kernel implementation MUST have
either:
  1. A non-gated fast path immediately following it (another early-return that
     does NOT require the unreachable feature), OR
  2. A `_TEST_HITS` counter on the fallback path proving that the slow path is
     monitored via the dispatch manifest.

If neither condition holds, the kernel has a silent performance cliff on
platforms where the feature is unavailable — exactly the defect that made Clip
76.8% of MobileNetV2 runtime (instance #14) and Conv scalar-only on macOS
(instance #8).

What this checks
----------------
For each Rust source file in the CPU EP kernel directory, this script:
  1. Finds `#[cfg(feature = "mlas")]` annotations on functions or blocks
     inside `fn execute(` implementations.
  2. Checks whether the code AFTER the gated block (still within the same
     `execute` function) contains either:
     - Another fast-path function call that is NOT behind cfg(feature = "mlas")
       (pattern: a function returning Result<bool> or using early return), OR
     - A `_TEST_HITS` counter increment (fetch_add) on the remaining path.
  3. Reports violations: gated fast paths whose fallback is unmonitored.

The check is deliberately conservative: it flags potential problems for human
review rather than trying to prove absence of fast paths through full control
flow analysis.

Relationship to other lints
----------------------------
  check_platform_naming.py       → file-level: single-arch file without name
  check_dispatch_reachability.py → counter-level: counter without paired test
  check_dispatch_manifest.py     → claim-level: optimization without counter
  check_feature_gate_coverage.py → gate-level: gated fast path with no fallback
                                    instrumentation (THIS SCRIPT)

Each lint declares its own blind spot. Together they form layered defense.

Known gaps (what this check CANNOT catch)
------------------------------------------
1. Performance paths gated by runtime feature detection (e.g.
   `is_x86_feature_detected!`) rather than compile-time cfg. Those paths are
   always compiled in and the gating is dynamic — a different class of problem
   requiring benchmarks, not lint.

2. A "fast path" that is actually slow. This check identifies structural
   patterns (cfg-gated early returns) but cannot measure actual performance.
   A well-instrumented but pathologically slow path passes this lint.

3. Feature gates in non-kernel code (optimizer passes, graph rewrites). The
   check scopes to `fn execute` implementations in the kernels directory.
   Optimizer-level gates that control which ops get fused are a separate
   concern (documented in dispatch_manifest.toml known gaps).

4. Gates using `cfg_attr` or `cfg!()` macro (runtime check). We scan for
   `#[cfg(` attribute patterns only. The `cfg!()` macro creates a runtime
   bool and the code is always compiled — different failure mode.

5. Newly added gates between lint runs. Like all static analysis, this runs
   at CI time. A developer can write a gated path and push — the lint catches
   it at PR, not at `cargo check` time. A pre-commit hook is the mitigation
   (see scripts/check_pre_commit.sh).

6. Exemptions. An [[exemption]] in `feature_gate_exemptions.toml` suppresses a
   violation with a stated justification. The exemption mechanism allows a
   human to assert "this fallback is acceptable" — but the human can be wrong.
   An exemption that becomes stale (e.g. the op moves from 1 node to 35 nodes
   in a new model) is invisible to this lint. Periodic review of the exemption
   file is the mitigation, not automation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

# ─── Configuration ──────────────────────────────────────────────────────────

# Features that are unreachable on specific platforms we ship on.
# Format: (feature_name, description_of_unreachability)
UNREACHABLE_FEATURES = [
    ("mlas", "mlas-sys is x86_64-linux-gnu only; unreachable on macOS/aarch64"),
]

# Directory containing kernel implementations.
KERNEL_DIR = Path("crates/onnx-runtime-ep-cpu/src/kernels")

# Exemption file: deliberate decisions not to optimize, with Amdahl reasoning.
EXEMPTION_FILE = Path("feature_gate_exemptions.toml")

# Pattern for cfg(feature = "X") where X is an unreachable feature.
CFG_FEATURE_PATTERN = re.compile(
    r'#\[cfg\((?:all\()?(?:.*,\s*)?feature\s*=\s*"(' +
    "|".join(re.escape(f) for f, _ in UNREACHABLE_FEATURES) +
    r')"'
)

# Pattern for a TEST_HITS counter increment (proves fallback is monitored).
TEST_HITS_PATTERN = re.compile(r'\w+_TEST_HITS\s*\.\s*fetch_add')

# Pattern for the start of an `execute` method implementation.
EXECUTE_FN_PATTERN = re.compile(r'fn\s+execute\s*\(')

# Pattern for a non-gated fast-path early return (function call returning
# Result<bool> with a `return Ok(())` after the `if`).
FAST_PATH_CALL_PATTERN = re.compile(
    r'if\s+\w+\([^)]*\)\?\s*\{?\s*\n\s*return\s+Ok\(\(\)\)'
)

# Pattern detecting a function defined behind cfg(feature = "mlas").
GATED_FN_PATTERN = re.compile(
    r'#\[cfg\((?:all\()?(?:.*,\s*)?feature\s*=\s*"(' +
    "|".join(re.escape(f) for f, _ in UNREACHABLE_FEATURES) +
    r')"\)?\)\]\s*\n\s*(?:pub(?:\(crate\))?\s+)?fn\s+(\w+)'
)


def find_execute_blocks(source: str) -> list[tuple[int, int]]:
    """Find (start_line, end_line) of each `fn execute` implementation."""
    lines = source.split('\n')
    blocks = []
    i = 0
    while i < len(lines):
        if EXECUTE_FN_PATTERN.search(lines[i]):
            # Find the end of this function by brace counting.
            start = i
            depth = 0
            found_open = False
            for j in range(i, len(lines)):
                for ch in lines[j]:
                    if ch == '{':
                        depth += 1
                        found_open = True
                    elif ch == '}':
                        depth -= 1
                if found_open and depth == 0:
                    blocks.append((start, j))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1
    return blocks


def find_gated_calls_in_execute(source: str, start: int, end: int) -> list[dict]:
    """Find cfg(feature="mlas") gated early-return calls within an execute block."""
    lines = source.split('\n')
    block_text = '\n'.join(lines[start:end + 1])
    
    # First, find all functions defined behind the unreachable feature gate.
    gated_fns = set()
    full_lines = source.split('\n')
    for match in GATED_FN_PATTERN.finditer(source):
        gated_fns.add(match.group(2))
    
    violations = []
    
    # Look for the pattern: #[cfg(feature = "mlas")] followed by a call or block
    # that provides an early return, where the subsequent code lacks monitoring.
    block_lines = lines[start:end + 1]
    
    for i, line in enumerate(block_lines):
        if not CFG_FEATURE_PATTERN.search(line):
            continue
        
        # Found a cfg(feature = "mlas") gate inside execute.
        # Check if this is an early-return pattern (the gated call returns early).
        # Look at the next few lines for a call pattern like:
        #   if mlas_fn(...)? { return Ok(()); }
        gate_line = start + i
        
        # Look ahead for the gated call/block pattern.
        lookahead = '\n'.join(block_lines[i:min(i + 8, len(block_lines))])
        if 'return' not in lookahead and 'Ok(())' not in lookahead:
            continue  # Not a fast-path early return, skip.
        
        # Now check: is there a non-gated fast path OR a TEST_HITS counter
        # in the remainder of this execute block (after the gated section)?
        remainder_start = i + 1
        # Skip past the gated block.
        depth = 0
        for j in range(i + 1, len(block_lines)):
            for ch in block_lines[j]:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            if depth <= 0 and j > i + 1:
                remainder_start = j + 1
                break
        
        remainder = '\n'.join(block_lines[remainder_start:])
        
        # Check for non-gated fast path (another function call with early return
        # that is NOT behind cfg(feature = "mlas")).
        has_non_gated_fast_path = False
        
        # Look for patterns like:
        #   if some_fn(...)? { return Ok(()); }
        # that are NOT preceded by #[cfg(feature = "mlas")]
        remainder_lines = block_lines[remainder_start:]
        for k, rline in enumerate(remainder_lines):
            # Check if this line has a fast-path call pattern.
            if re.search(r'if\s+\w+\(', rline) and '?' in rline:
                # Check that the preceding lines don't have a mlas cfg gate.
                preceding = '\n'.join(remainder_lines[max(0, k-3):k+1])
                if not CFG_FEATURE_PATTERN.search(preceding):
                    has_non_gated_fast_path = True
                    break
            # Also check for direct platform-gated paths (macOS/NEON).
            if re.search(r'#\[cfg\((?:all\()?(?:any\()?target_(?:os|arch)', rline):
                # A platform-specific path exists — it's reachable on that platform.
                has_non_gated_fast_path = True
                break
        
        # Check for TEST_HITS counter.
        has_test_hits = bool(TEST_HITS_PATTERN.search(remainder))
        
        if not has_non_gated_fast_path and not has_test_hits:
            violations.append({
                'line': gate_line + 1,  # 1-indexed
                'feature': 'mlas',
            })
    
    return violations


def check_file(path: Path) -> list[dict]:
    """Check a single file for unmonitored feature-gated fast paths.
    
    Returns a list of violation dicts with 'file', 'line', 'feature' keys.
    """
    source = path.read_text()
    results = []
    
    execute_blocks = find_execute_blocks(source)
    if not execute_blocks:
        return results
    
    for start, end in execute_blocks:
        violations = find_gated_calls_in_execute(source, start, end)
        for v in violations:
            results.append({
                'file': str(path),
                'line': v['line'],
                'feature': v['feature'],
            })
    
    return results


# ─── Exemption mechanism ────────────────────────────────────────────────────
#
# Modelled on dispatch_manifest.toml's [[exclusion]] semantics. Each exemption
# is a deliberate decision not to optimize, carrying Amdahl reasoning. An
# exemption without a justification is itself a lint failure — "we chose not
# to" must be distinguishable from "we forgot".


def load_exemptions(path: Path) -> tuple[dict[tuple[str, int], dict], list[str]]:
    """Load exemptions from TOML file.
    
    Returns (exemptions_dict, errors).
    exemptions_dict maps (file, line) -> exemption record.
    errors is a list of validation failures in the exemption file itself.
    """
    if not path.exists():
        return {}, []
    
    if tomllib is None:
        # Minimal inline TOML parser for the subset we use (array of tables
        # with string/int fields). Matches check_dispatch_manifest.py pattern.
        return _parse_exemptions_fallback(path)
    
    with open(path, "rb") as f:
        data = tomllib.load(f)
    
    exemptions: dict[tuple[str, int], dict] = {}
    errors: list[str] = []
    
    for i, ex in enumerate(data.get("exemption", [])):
        # Validate required fields.
        file_field = ex.get("file")
        line = ex.get("line")
        feature = ex.get("feature")
        reason = ex.get("reason")
        
        if not file_field:
            errors.append(f"[[exemption]] #{i+1}: missing 'file' field")
            continue
        if not line:
            errors.append(f"[[exemption]] #{i+1} ({file_field}): missing 'line' field")
            continue
        if not feature:
            errors.append(f"[[exemption]] #{i+1} ({file_field}:{line}): missing 'feature' field")
            continue
        if not reason or len(reason.strip()) < 20:
            errors.append(
                f"[[exemption]] #{i+1} ({file_field}:{line}): 'reason' must be a "
                f"substantive justification (≥20 chars). An exemption without Amdahl "
                f"reasoning is indistinguishable from 'we forgot'. Got: {reason!r}"
            )
            continue
        
        exemptions[(file_field, int(line))] = ex
    
    return exemptions, errors


def _parse_exemptions_fallback(path: Path) -> tuple[dict[tuple[str, int], dict], list[str]]:
    """Minimal TOML parser for [[exemption]] arrays when tomllib is unavailable."""
    text = path.read_text()
    exemptions: dict[tuple[str, int], dict] = {}
    errors: list[str] = []
    current: dict | None = None
    idx = 0

    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#') or not stripped:
            continue
        if stripped == '[[exemption]]':
            if current is not None:
                _validate_and_add(current, idx, exemptions, errors)
            current = {}
            idx += 1
            continue
        if current is not None and '=' in stripped:
            key, _, val = stripped.partition('=')
            key = key.strip()
            val = val.strip()
            if val.startswith('"') and val.endswith('"'):
                current[key] = val[1:-1]
            elif val.isdigit():
                current[key] = int(val)
            else:
                current[key] = val

    if current is not None:
        _validate_and_add(current, idx, exemptions, errors)

    return exemptions, errors


def _validate_and_add(
    ex: dict, idx: int,
    exemptions: dict[tuple[str, int], dict],
    errors: list[str],
) -> None:
    """Validate a single exemption record and add to dict or errors."""
    file_field = ex.get("file")
    line = ex.get("line")
    feature = ex.get("feature")
    reason = ex.get("reason")

    if not file_field:
        errors.append(f"[[exemption]] #{idx}: missing 'file' field")
        return
    if not line:
        errors.append(f"[[exemption]] #{idx} ({file_field}): missing 'line' field")
        return
    if not feature:
        errors.append(f"[[exemption]] #{idx} ({file_field}:{line}): missing 'feature' field")
        return
    if not reason or len(str(reason).strip()) < 20:
        errors.append(
            f"[[exemption]] #{idx} ({file_field}:{line}): 'reason' must be a "
            f"substantive justification (≥20 chars). An exemption without Amdahl "
            f"reasoning is indistinguishable from 'we forgot'. Got: {reason!r}"
        )
        return

    exemptions[(str(file_field), int(line))] = ex


def main() -> int:
    if not KERNEL_DIR.exists():
        print(f"ERROR: kernel directory not found: {KERNEL_DIR}", file=sys.stderr)
        return 1
    
    # Load exemptions.
    exemptions, exemption_errors = load_exemptions(EXEMPTION_FILE)
    if exemption_errors:
        print("=" * 72)
        print("FEATURE GATE COVERAGE: exemption file has errors")
        print("=" * 72)
        print()
        for err in exemption_errors:
            print(f"  ✗ {err}")
            print()
        return 1
    
    # Collect all violations.
    all_violations: list[dict] = []
    for rs_file in sorted(KERNEL_DIR.glob("*.rs")):
        all_violations.extend(check_file(rs_file))
    
    # Separate exempted from unexempted.
    unexempted: list[dict] = []
    exempted: list[dict] = []
    for v in all_violations:
        key = (v['file'], v['line'])
        if key in exemptions:
            exempted.append(v)
        else:
            unexempted.append(v)
    
    # Check for stale exemptions (exemptions that don't match any violation).
    stale: list[tuple[str, int]] = []
    violation_keys = {(v['file'], v['line']) for v in all_violations}
    for key in exemptions:
        if key not in violation_keys:
            stale.append(key)
    
    has_failures = bool(unexempted) or bool(stale)
    
    if has_failures:
        print("=" * 72)
        print("FEATURE GATE COVERAGE: violations detected")
        print("=" * 72)
        print()
        
        if unexempted:
            for v in unexempted:
                print(
                    f"  ✗ {v['file']}:{v['line']}: cfg(feature = \"{v['feature']}\") "
                    f"guards a fast path in `execute` but the fallback has no "
                    f"non-gated fast path and no _TEST_HITS counter. The optimization "
                    f"is unreachable on platforms where '{v['feature']}' is unavailable "
                    f"(macOS/aarch64). Add a platform-native fast path or instrument "
                    f"the fallback with a _TEST_HITS counter so the dispatch manifest "
                    f"can track it."
                )
                print()
            print(f"{len(unexempted)} unexempted violation(s).")
            print()
            print("Why this matters: cfg(feature = \"mlas\") is unreachable on macOS/")
            print("aarch64 (mlas-sys is x86_64-linux-gnu only). Without a non-gated")
            print("fast path or a monitored fallback, the op silently degrades to a")
            print("pathologically slow scalar implementation on Apple Silicon.")
            print()
            print("Fix options:")
            print("  1. Add a platform-native fast path (NEON/vDSP/Accelerate)")
            print("  2. Add a _TEST_HITS counter on the fallback and a manifest row")
            print("     declaring the expected tier (even if tier3/scalar)")
            print("  3. If the fallback is genuinely acceptable (Amdahl: negligible")
            print("     share of runtime), add an [[exemption]] to")
            print(f"     {EXEMPTION_FILE} with substantive reasoning.")
            print()
        
        if stale:
            print("  Stale exemptions (no longer match a violation — remove them):")
            for file, line in stale:
                ex = exemptions[(file, line)]
                print(f"    • {file}:{line} (feature={ex.get('feature')})")
            print()
        
        return 1
    
    # Success output.
    if exempted:
        print(f"✓ All feature-gated fast paths have fallback coverage.")
        print(f"  ({len(exempted)} deliberate exemption(s) acknowledged:)")
        for v in exempted:
            key = (v['file'], v['line'])
            ex = exemptions[key]
            print(f"    • {v['file']}:{v['line']} — {ex.get('reason', '?')[:60]}")
    else:
        print("✓ All feature-gated fast paths have fallback coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
