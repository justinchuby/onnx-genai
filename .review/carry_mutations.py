#!/usr/bin/env python3
"""Mutation battery for the workflow-session carry accounting.

Each arm is a real wrong implementation, not a syntactic tweak. Restores
rewrite the file rather than copying one back: a preserved older mtime makes
cargo skip the rebuild, and a skipped rebuild reads as a false SURVIVED.
"""

import hashlib
import subprocess
import sys

WORKFLOW_API = "crates/onnx-genai-engine/src/engine/workflow_api.rs"
RUNTIME = "crates/onnx-genai-engine/src/engine/runtime.rs"

TEST = "engine::runtime::tests::a_continuing_turn_is_admitted_for_the_conversation_it_carries"

MUTATIONS = [
    (
        "M7 session-reuse arm unreachable (every turn mints a fresh scheduler id)",
        WORKFLOW_API,
        "            Some(session_id) if self.workflow_sessions.contains_key(&session_id) => session_id,",
        "            Some(session_id) if false && self.workflow_sessions.contains_key(&session_id) => session_id,",
    ),
    (
        "M8 carry always reads 0 (the id is reused, the prefix is not charged)",
        RUNTIME,
        "        let carried = if self.workflow_sessions.contains_key(&session_id) {",
        "        let carried = if false && self.workflow_sessions.contains_key(&session_id) {",
    ),
    (
        "M9 vacuity: admission refuses every workflow-driven turn",
        RUNTIME,
        "        let prompt_tokens = self.interpreted_prompt_token_count(prompt)? + carried;",
        "        let prompt_tokens = self.interpreted_prompt_token_count(prompt)? + carried + 1_000_000;",
    ),
]


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def run_suite():
    proc = subprocess.run(
        [
            "taskset", "-c", "8-15",
            "cargo", "test", "-p", "onnx-genai-engine",
            "--features", "native-backend", "--lib", TEST, "--", "--exact",
        ],
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    # A filter that matches nothing exits 0 and reads exactly like a survived
    # mutation. Every arm below would have been reported as SURVIVED -- the
    # vacuity arm included -- so the count is checked, not the exit code alone.
    if "1 passed" not in out and "1 failed" not in out:
        sys.exit(f"FATAL: the filter selected no test, so nothing was measured\n{out[-2000:]}")
    return proc.returncode, out


def apply(path, old, new):
    s = open(path).read()
    if s.count(old) != 1:
        sys.exit(f"FATAL: anchor not unique in {path} ({s.count(old)} matches)\n  {old}")
    open(path, "w").write(s.replace(old, new))


def main():
    before = {p: digest(p) for p in (WORKFLOW_API, RUNTIME)}

    code, out = run_suite()
    print(f"baseline: {'PASS' if code == 0 else 'FAIL'}")
    if code != 0:
        print(out[-3000:])
        sys.exit("FATAL: baseline must be green before mutating")

    results = []
    for label, path, old, new in MUTATIONS:
        apply(path, old, new)
        code, out = run_suite()
        caught = code != 0
        results.append((label, caught))
        print(f"\n{'=' * 78}\n{label}\n  -> {'CAUGHT' if caught else '** SURVIVED **'}")
        if caught:
            for line in out.splitlines():
                if any(k in line for k in ("was admitted", "was refused", "on_admitted()", "must be visible")):
                    print(f"     {line.strip()}")
        # Restore by rewriting, so the mtime advances and cargo rebuilds.
        s = open(path).read()
        open(path, "w").write(s.replace(new, old))

    after = {p: digest(p) for p in (WORKFLOW_API, RUNTIME)}
    print(f"\n{'=' * 78}")
    for p in before:
        state = "identical" if before[p] == after[p] else "** DRIFTED **"
        print(f"tree {p.split('/')[-1]}: {state} ({before[p]})")

    code, _ = run_suite()
    print(f"post-battery baseline: {'PASS' if code == 0 else '** FAIL **'}")

    survived = [label for label, caught in results if not caught]
    print(f"\n{len(results) - len(survived)}/{len(results)} caught")
    if survived or code != 0 or before != after:
        sys.exit(1)


if __name__ == "__main__":
    main()
