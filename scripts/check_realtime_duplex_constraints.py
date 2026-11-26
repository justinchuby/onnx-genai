#!/usr/bin/env python3
"""Reproduce the Native duplex counterexample, then run exact fixed regressions.

Run under the shared host workload lock. This driver does not acquire that lock.
It reuses conformance process-group cleanup and requires nonempty libtest discovery.
The baseline worktree must be detached at the authored counterexample commit;
the fixed worktree may contain the subsequently integrated/cherry-picked changes.
Logs bind clean HEAD/tree, selected source/manifests/lock hashes and effective
Cargo configuration/environment before and after every command. Successful runs
write verified.json; these checks do not claim whole-goal parity or hash arbitrary
external compiler/executable contents.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from audit_llama_server_conformance import (
    cargo_context,
    digest,
    listed_test,
    passed_test,
    run_contained,
    source_state,
)

BASELINE_REVISION = "6b0b424a8fead460b5fc59e351bda3db4943d3ba"
WS = "realtime::websocket::tests::"
BASELINE = WS + "websocket_duplex_masks_higher_ranked_invalid_tool_before_first_audio_selection"
FIXED = [
    BASELINE,
    WS + "websocket_duplex_freezes_tool_plan_across_updates_and_rearms_after_cancel",
    WS + "websocket_duplex_constraint_progress_survives_poll_without_rearming",
    WS + "websocket_duplex_rejects_missing_mask_and_unsupported_forcing_before_audio",
    WS + "native_owner_arms_generic_constraint_without_consuming_a_quantum",
    "realtime::tools::tests::active_catalog_shares_immutable_schemas_while_defaults_are_replaced",
    WS + "websocket_duplex_masks_early_eos_and_admits_declared_eos_after_tool_completion",
    WS + "websocket_duplex_rejects_native_eos_when_graph_ignores_required_mask",
    WS + "websocket_native_turn_limit_truncates_tool_prefix_then_override_reaches_eos",
]


def source_identity(
    tree: Path, config: Path | None, environment: dict[str, str], *, baseline: bool = False
) -> dict:
    state = source_state(tree)
    head = subprocess.run(
        ["git", "-C", str(tree), "symbolic-ref", "--quiet", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if head.returncode not in (0, 1):
        raise RuntimeError(f"cannot inspect symbolic HEAD in {tree}: {head.stderr.strip()}")
    symbolic_head = head.stdout.strip() if head.returncode == 0 else None
    if baseline and (state["revision"] != BASELINE_REVISION or symbolic_head is not None):
        raise ValueError(f"baseline tree must be clean and detached at {BASELINE_REVISION}")
    return {
        "worktree": str(tree),
        **state,  # Clean HEAD/tree binds all tracked source, not just selected files.
        "symbolic_head": symbolic_head,
        "source_sha256": {
            name: digest(tree / name)
            for name in [
                "Cargo.toml", "Cargo.lock",
                "crates/onnx-genai-server/Cargo.toml",
                "crates/onnx-genai-engine/Cargo.toml",
                "crates/onnx-genai-server/src/realtime.rs",
                "crates/onnx-genai-server/src/realtime/tools.rs",
                "crates/onnx-genai-server/src/realtime/websocket.rs",
                "crates/onnx-genai-engine/src/realtime.rs",
                "crates/onnx-genai-engine/src/byte_grammar.rs",
            ]
        },
        # Includes implicit/explicit config, include chains, absent-file sentinels,
        # and hashes of the full environment plus relevant build variable values.
        "cargo_context": cargo_context(tree, config, environment),
    }


def require_unchanged(expected: dict, current: dict) -> None:
    if current != expected:
        raise ValueError("source/HEAD/Cargo configuration/environment changed during validation; discard this run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tree", type=Path)
    parser.add_argument("--fixed-tree", type=Path, required=True)
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--cargo-config", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error("--jobs and --timeout must be positive")
    if not args.fixed_only and args.baseline_tree is None:
        parser.error("provide --baseline-tree or select --fixed-only")
    fixed = args.fixed_tree.resolve(strict=True)
    baseline = args.baseline_tree.resolve(strict=True) if not args.fixed_only else None
    args.log_dir = args.log_dir.resolve()
    if any(args.log_dir.is_relative_to(tree) for tree in [fixed, *([baseline] if baseline else [])]):
        parser.error("--log-dir must be outside both source worktrees")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="duplex-", dir=args.log_dir))
    command = [
        "cargo", "test", "-p", "onnx-genai-server", "--lib", "-j", str(args.jobs),
        "--no-default-features", "--features", "metrics,native-backend,cuda-13000",
    ]
    if args.offline:
        command.append("--offline")
    config = args.cargo_config.resolve(strict=True) if args.cargo_config else None
    if config:
        command += ["--config", str(config)]
    target_dir = str(args.target_dir.resolve())

    def capture_inputs() -> dict:
        environment = dict(os.environ, CARGO_TARGET_DIR=target_dir)
        return {
            "fixed": source_identity(fixed, config, environment),
            "baseline": source_identity(baseline, config, environment, baseline=True) if baseline else None,
        }

    inputs = capture_inputs()
    environment = dict(os.environ, CARGO_TARGET_DIR=target_dir)
    metadata = {
        "command": command,
        "target_dir": environment["CARGO_TARGET_DIR"],
        "inputs": inputs,
        "expected_fixed_cases": FIXED,
    }
    (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Native duplex validation logs: {output}", flush=True)

    def execute(tree: Path, selector: str, label: str, listing: bool = False):
        invocation = command + [selector, "--", "--exact", "--nocapture"]
        if listing:
            invocation.append("--list")
        path = output / f"{label}.log"
        require_unchanged(inputs, capture_inputs())
        try:
            with path.open("w") as log:
                result = run_contained(
                    invocation, cwd=tree, stdout=log, timeout=args.timeout, env=environment
                )
        finally:
            # run_contained first terminates/reaps descendants, then this check
            # refuses evidence if either worktree/config/environment changed.
            require_unchanged(inputs, capture_inputs())
        return result.returncode, path.read_text()

    if baseline:
        code, text = execute(baseline, BASELINE, "baseline-discovery", True)
        if code:
            raise RuntimeError("baseline discovery/build failed; inspect baseline-discovery.log")
        listed_test(text, BASELINE)
        code, text = execute(baseline, BASELINE, "baseline-expected-failure")
        if (
            code == 0
            or "duplex must mask Rome before Native selection" not in text
            or not re.search(r"test result: FAILED\. 0 passed; 1 failed; 0 ignored;", text)
            or not re.search(r"^test " + re.escape(BASELINE) + r" \.\.\. FAILED$", text, re.MULTILINE)
        ):
            raise RuntimeError("baseline did not fail its intended Native mask assertion; inspect its log")
        print("Baseline: exact Native mask counterexample failed as expected", flush=True)
    for index, selector in enumerate(FIXED, 1):
        code, text = execute(fixed, selector, f"fixed-{index:02}-discovery", True)
        if code:
            raise RuntimeError(f"{selector}: discovery/build failed; inspect logs")
        listed_test(text, selector)
        code, text = execute(fixed, selector, f"fixed-{index:02}-run")
        if code:
            raise RuntimeError(f"{selector}: execution failed; inspect logs")
        passed_test(text, selector)
        print(f"Fixed {index}/{len(FIXED)}: 1 passed: {selector}", flush=True)
    final_inputs = capture_inputs()
    require_unchanged(inputs, final_inputs)
    (output / "verified.json").write_text(json.dumps({
        "status": "passed",
        "baseline_expected_failure": baseline is not None,
        "fixed_passed": len(FIXED),
        "inputs": final_inputs,
    }, indent=2) + "\n")
    print(f"Verified {len(FIXED)} exact Native regression cases; logs: {output}", flush=True)


if __name__ == "__main__":
    main()
