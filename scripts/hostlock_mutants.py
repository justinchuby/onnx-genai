#!/usr/bin/env python3
"""Mutation harness for the host-lock reader and the measurement window.

Run: `python3 scripts/hostlock_mutants.py`

`hostlock.rs` reads the advisory host lock that `scripts/hostlock.sh` writes,
and turns it into the `host_lock=` field carried by every benchmark result row.
Its whole job is to refuse to certify a measurement, so its failures are silent
in the direction that permits a run: a reader that classified a held lock as
`free`, or a stranger's lock as `mine`, would publish a confident row and
nothing downstream would notice.

Tests can have the same failure mode as the thing they test. A test that would
still pass with the defect reintroduced is a claim in a doc comment, not a
check. So each entry below is one defect the module is supposed to be protected
against, applied to the real source; a mutant no test kills is a promise
nothing keeps.

Every mutant here corresponds to a rule that was arrived at the hard way --
either from `hostlock.sh`'s own hazard notes, from a review find, or from a
test that failed the first time it was run against the real script.

Two guards, both learned by being caught out:

* The run counts are captured from the baseline and then required to match on
  every mutant run. A mis-targeted filter, or a compile error in a later test
  target, otherwise reads as a clean pass. `--no-fail-fast` is not optional:
  cargo stops at the first failing target, so without it the integration tests
  never run under a mutant that kills a unit test, and their absent count looks
  like a mis-targeted filter.

* The source is restored on every exit path, including a signal. Half of these
  mutants make the module certify measurements it should refuse; leaving one in
  a working tree is worse than never having run this.
"""

import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOSTLOCK = ROOT / "crates/onnx-runtime-hostmon/src/hostlock.rs"
WINDOW = ROOT / "crates/onnx-runtime-hostmon/src/window.rs"

# (file, name, text to replace, replacement). The text must match the source
# exactly; a mutant whose anchor has drifted is reported rather than skipped,
# because a mutant that never applied is indistinguishable from one that was
# killed.
MUTANTS = [
    # --- owner strings are untrusted input, and end up in a `key=value` row ---
    (HOSTLOCK, "owner-passthrough",
     "            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {",
     "            if true {"),
    (HOSTLOCK, "owner-unbounded",
     "        .take(MAX_OWNER_LEN)\n",
     ""),
    (HOSTLOCK, "owner-empty-blank",
     '        "?".to_string()',
     '        String::new()'),

    # --- the metadata file must be read the way the script writes it ---
    (HOSTLOCK, "meta-key-substring",
     "        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))",
     "        .find_map(|line| line.split_once('=').filter(|(k, _)| k.contains(key)).map(|(_, v)| v))"),
    (HOSTLOCK, "meta-last-wins",
     "    meta.lines()\n        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))",
     "    meta.lines()\n        .filter_map(|line| line.strip_prefix(key)?.strip_prefix('='))\n        .last()"),

    # --- liveness, which must agree with hostlock.sh line for line ---
    (HOSTLOCK, "no-pid-is-alive",
     "    let Some(pid) = holder.anchor_pid else {\n        return Liveness::Unprovable;\n    };",
     "    let Some(pid) = holder.anchor_pid else {\n        return Liveness::Alive;\n    };"),
    (HOSTLOCK, "zombie-ignored",
     "    if info.state == 'Z' && info.threads.is_some_and(|t| t <= 1) {\n        return Liveness::Dead;\n    }\n",
     ""),
    (HOSTLOCK, "zombie-leader-reaped",
     "    if info.state == 'Z' && info.threads.is_some_and(|t| t <= 1) {",
     "    if info.state == 'Z' {"),
    (HOSTLOCK, "start-time-ignored",
     "        Some(recorded) if recorded != info.start_time => Liveness::Dead,",
     "        Some(_recorded) if false => Liveness::Dead,"),
    (HOSTLOCK, "missing-start-is-alive",
     "        None => Liveness::Unprovable,\n    }\n}",
     "        None => Liveness::Alive,\n    }\n}"),
    (HOSTLOCK, "departed-pid-is-alive",
     "        // No `/proc` entry at all is the one unambiguous death.\n        return Liveness::Dead;",
     "        return Liveness::Alive;"),

    # --- what the field is allowed to claim ---
    (HOSTLOCK, "no-two-ended-read",
     "    if before != after {\n        return LockField::Changed;\n    }",
     ""),
    (HOSTLOCK, "foreign-is-mine",
     "            Some(_) => LockField::Foreign(holder.owner.clone()),",
     "            Some(_) => LockField::Mine(holder.owner.clone()),"),
    (HOSTLOCK, "unattributed-is-mine",
     "            None => LockField::Held(holder.owner.clone()),",
     "            None => LockField::Mine(holder.owner.clone()),"),
    (HOSTLOCK, "held-protects",
     "        matches!(self, LockField::Mine(_))",
     "        matches!(self, LockField::Mine(_) | LockField::Held(_))"),
    (HOSTLOCK, "unverified-protects",
     "        matches!(self, LockField::Mine(_))",
     "        matches!(self, LockField::Mine(_) | LockField::Unverified(_))"),
    (HOSTLOCK, "stale-protects",
     "        matches!(self, LockField::Mine(_))",
     "        matches!(self, LockField::Mine(_) | LockField::Stale(_))"),
    (HOSTLOCK, "unprovable-is-held-and-stale-is-free",
     "        Liveness::Unprovable => LockState::Unverified(holder),",
     "        Liveness::Unprovable => LockState::Held(holder),"),
    (HOSTLOCK, "unparseable-is-free",
     "    let Some(holder) = parse_meta(meta) else {\n        return LockState::Unknown;\n    };",
     "    let Some(holder) = parse_meta(meta) else {\n        return LockState::Free;\n    };"),
    (HOSTLOCK, "io-error-is-free",
     "        Err(std::io::ErrorKind::NotFound) => LockState::Free,\n        Err(_) => LockState::Unknown,",
     "        Err(_) => LockState::Free,"),

    # --- attribution: the only value that certifies a row is `mine` ---
    (HOSTLOCK, "attribute-on-sanitised",
     "            Some(mine) if mine == holder.owner_raw => LockField::Mine(holder.owner.clone()),",
     "            Some(mine) if sanitise_owner(mine) == holder.owner => LockField::Mine(holder.owner.clone()),"),
    (HOSTLOCK, "owner-raw-sanitised",
     "        owner_raw: owner.trim().to_string(),",
     "        owner_raw: sanitise_owner(owner),"),
    (HOSTLOCK, "blank-owners-match",
     "            Some(mine) if mine.is_empty() || holder.owner_raw.is_empty() => {\n                LockField::Foreign(holder.owner.clone())\n            }\n",
     ""),
    (HOSTLOCK, "stale-is-free",
     "        Liveness::Dead => LockState::Stale(holder),",
     "        Liveness::Dead => LockState::Free,"),

    # --- the reader must look where the script writes ---
    (HOSTLOCK, "default-dir-drift",
     'pub const DEFAULT_LOCK_DIR: &str = "/tmp/onnx-genai-hostlock";',
     'pub const DEFAULT_LOCK_DIR: &str = "/tmp/onnx-genai-hostlock-moved";'),

    # --- the window: a claim about a span, not an instant ---
    (WINDOW, "window-one-ended",
     "            field: hostlock::field(&self.before, &after, self_owner),",
     "            field: hostlock::field(&after, &after, self_owner),"),
    (WINDOW, "window-reason-from-second-read",
     "            reason: hostlock::reason(&self.before, &after),",
     "            reason: hostlock::reason(&after, &after),"),
    (WINDOW, "window-close-ignores-the-second-read",
     "        let after = read_after();",
     "        let after = self.before.clone();\n        let _ = read_after;"),
    (WINDOW, "window-warns-on-everything",
     "        if self.is_protected() {\n            return None;\n        }",
     ""),
    (WINDOW, "window-never-warns",
     "        if self.is_protected() {\n            return None;\n        }",
     "        return None;"),
    (WINDOW, "window-drops-the-reason",
     "            Some(reason) => write!(f, \"host_lock={} lock_reason={reason}\", self.field),",
     "            Some(_) => write!(f, \"host_lock={}\", self.field),"),
    (WINDOW, "window-reason-accessor-blind",
     "        self.reason.as_deref()",
     "        None"),

    # --- the verdict cannot be fabricated ---
    # Killed by the `compile_fail` doctests on `Report`, which are the only
    # thing that can assert an API shape. Here for the same reason the doctests
    # carry a positive control: rustdoc does not enforce the error code, so
    # `compile_fail` alone would also pass if the block stopped compiling for a
    # reason nobody intended. If `Report`'s fields are ever renamed this anchor
    # stops applying, and an un-applied anchor is reported as a survivor rather
    # than skipped.
    (WINDOW, "window-report-fields-public",
     "pub struct Report {\n    field: LockField,\n    reason: Option<String>,",
     "pub struct Report {\n    pub field: LockField,\n    pub reason: Option<String>,"),

    # --- the two functions every benchmark actually calls ---
    # Nothing in-process can reach these: they read `HOSTLOCK_DIR` and
    # `HOSTLOCK_OWNER` from the environment. They are killed by the child-process
    # probe in `agrees_with_hostlock_sh.rs`, and before that probe existed all
    # three of these survived while the suite reported 31/31 -- the coverage was
    # of `close_as`, which no benchmark calls.
    (WINDOW, "window-open-does-not-read",
     "            before: hostlock::read(),",
     "            before: LockState::Unknown,"),
    (WINDOW, "window-close-does-not-read",
     "        self.close_as(owner.as_deref(), hostlock::read)",
     "        self.close_as(owner.as_deref(), || LockState::Unknown)"),
    (WINDOW, "window-close-forgets-the-owner",
     '        let owner = std::env::var("HOSTLOCK_OWNER").ok();',
     "        let owner: Option<String> = None;"),
    (WINDOW, "window-warning-forgets-the-owner",
     '            self.self_owner.as_deref().unwrap_or("unset")',
     '            "unset"'),

]


def run_tests():
    """Returns (verdict, detail). Verdict is compile-error, wrong-run-count,
    killed, or survived."""
    out = subprocess.run(
        ["cargo", "test", "-p", "onnx-runtime-hostmon", "--no-fail-fast", "--", "--quiet"],
        cwd=ROOT, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    if "error[" in text or "error: could not compile" in text:
        return "compile-error", text
    counts = [int(m) for m in re.findall(r"running (\d+) tests", text)]
    if out.returncode != 0:
        names = re.findall(r"^    ([\w:]+)$", text, re.M)
        # Doctest failures are named by path and line rather than by an
        # identifier, so the identifier pattern above misses them entirely and
        # the kill reads as `unnamed`. `window-report-fields-public` is killed
        # only by a doctest, and a kill nobody can attribute is halfway back to
        # not knowing whether the test ran.
        names += re.findall(r"^    (\S+\.rs - \S+ \(line \d+\))$", text, re.M)
        return "killed", (counts, ", ".join(sorted(set(names))) or "unnamed")
    return "survived", (counts, text)


def main():
    originals = {path: path.read_text() for path in (HOSTLOCK, WINDOW)}
    restored = False

    def restore(*_):
        nonlocal restored
        if not restored:
            for path, text in originals.items():
                path.write_text(text)
            restored = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (restore(), sys.exit(130)))

    verdict, detail = run_tests()
    if verdict != "survived":
        print(f"BASELINE NOT GREEN: {verdict}\n{detail}"[-3000:])
        return 1
    baseline_counts, _ = detail
    if not baseline_counts:
        print("BASELINE NOT GREEN: no test targets ran")
        return 1
    print(f"baseline green, test counts per target: {baseline_counts}\n")

    survivors = []
    try:
        for path, name, old, new in MUTANTS:
            original = originals[path]
            if old not in original:
                # Not a pass. `cargo fmt` reflowing an anchor has silently
                # un-applied a mutant here before, and an un-applied mutant is
                # indistinguishable from a killed one in the exit code.
                print(f"  !! {path.name}:{name}: anchor text not found -- mutant never applied")
                survivors.append(name + " (unapplied)")
                continue
            path.write_text(original.replace(old, new, 1))
            verdict, detail = run_tests()
            path.write_text(original)
            if verdict in ("killed", "survived"):
                counts, info = detail
                if counts != baseline_counts:
                    print(f"  !! {name}: ran {counts}, baseline {baseline_counts} -- not comparable")
                    survivors.append(name + " (run-count drift)")
                    continue
            if verdict == "killed":
                print(f"  killed   {name:38s} by {detail[1]}")
            else:
                print(f"  SURVIVED {name:38s} ({verdict})")
                survivors.append(name)
    finally:
        restore()

    if any(path.read_text() != text for path, text in originals.items()):
        print("\n!! source not restored -- check `git diff` before doing anything else")
        return 2
    if survivors:
        print(f"\n{len(survivors)} survivor(s): {survivors}")
        return 1
    print(f"\nall {len(MUTANTS)} mutants killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
