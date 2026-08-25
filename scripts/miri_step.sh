#!/usr/bin/env bash
# miri_step.sh — the Miri lane's step wrapper, with an arity guard
#
# Sourced by .github/workflows/miri.yml. It exists as a file rather than a
# function inside the workflow so that it can have a conformance suite:
# `run_miri` is a gate whose failure mode is a GREEN report, and a gate like
# that cannot be reviewed by reading it. See scripts/miri_step_test.sh.
#
# ── Why the arity guard ────────────────────────────────────────────────────
#
# Most of the Miri lane selects tests BY NAME. Of its 35 invocations, 32 carry
# a module-path or exact-name filter and 14 of those match exactly ONE test --
# each one carrying a specific soundness argument that its surrounding comment
# spells out ("removing the `catch_unwind` reports a data race", and so on).
#
# libtest treats a filter that matches nothing as success:
#
#     $ cargo test --lib -- a_name_that_does_not_exist; echo $?
#     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out
#     0
#
# So renaming, moving, or adding `#[cfg_attr(miri, ignore)]` to a selected test
# turns its step into a no-op that reports `ok` and exits 0. `set -euo pipefail`
# does not catch it, because there is no error: the step really did succeed at
# running nothing. The soundness argument stops being checked and the only
# evidence is a line in a folded log group that nobody reads while it is green.
#
# Note that this asymmetry is not libtest being careless -- an unmatched filter
# is normal and correct when a human types one. It is only a defect when the
# filter is a durable assertion about what CI covers, which is exactly what
# these 32 are. A wrong `-p` IS loud (`cargo` exits 101, "did not match any
# packages"); a wrong test filter is not. The guard restores the symmetry.
#
# ── What counts as "ran" ───────────────────────────────────────────────────
#
# Executed means passed + failed >= 1, summed across every `test result:` line
# the step printed. Ignored deliberately does NOT count. An all-ignored step is
# precisely the case worth failing on: it is how a step silently stops checking
# its property while still looking like it runs. If a step is intentionally
# all-ignored under Miri, delete the step -- there is no opt-out here, because
# an opt-out is the hole this closes.
#
# Summing across lines rather than requiring every line to be non-empty is
# deliberate: a single cargo invocation prints one `test result:` per target,
# and auxiliary targets legitimately report `0 passed; 0 failed`.

# Run one Miri step, streaming its output, and fail loudly if it ran no tests.
#
# Usage: run_miri "<label>" <command...>
run_miri() {
  local label="$1"
  shift
  local start end rc executed log

  # Alongside the checkout rather than in TMPDIR: on a failure the log is the
  # artifact you want, and a runner's temp dir is not always collected.
  log="${MIRI_STEP_LOG_DIR:-.}/.miri_step.log"

  start=$(date +%s)
  echo "::group::Miri: ${label}"

  # `tee` keeps the output streaming -- these steps run for minutes and a
  # captured-then-printed log makes a hang indistinguishable from a slow pass.
  # The status therefore has to come from PIPESTATUS: plain `$?` after a
  # pipeline is `tee`'s status, and `tee` succeeds at copying a failure.
  set +e
  "$@" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e

  end=$(date +%s)
  # Emitted before any early return so a failing step still closes its log
  # group and still reports its duration.
  echo "MIRI_TIMING ${label}: $((end - start))s"
  echo "::endgroup::"

  if [ "${rc}" -ne 0 ]; then
    echo "::error::Miri step '${label}' failed (exit ${rc})."
    return "${rc}"
  fi

  executed=$(awk '
    /^test result:/ {
      for (i = 1; i < NF; i++) {
        if ($(i + 1) == "passed;") { p += $i }
        if ($(i + 1) == "failed;") { f += $i }
      }
    }
    END { print p + f + 0 }
  ' "${log}")

  if [ "${executed}" -eq 0 ]; then
    echo "::error::Miri step '${label}' executed 0 tests and still reported success. \
Its filter matches nothing -- the selected test was probably renamed, moved, or \
marked #[cfg_attr(miri, ignore)]. This step has been asserting nothing. Point it \
at the test's new name, or delete the step; do not leave it green."
    return 1
  fi

  echo "MIRI_EXECUTED ${label}: ${executed}"
}
