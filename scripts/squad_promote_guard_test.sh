#!/usr/bin/env bash
#
# Self-test for scripts/squad_promote_guard.sh.
#
# The defect these guards had is one where the broken instrument returns the
# same value as the healthy one, so the only test that means anything is a
# *differential* one: every arm below that the fixed form fails is also run
# against the pre-fix form, and the pre-fix form must pass it. An arm both
# forms fail, or both pass, discriminates nothing and is not evidence.
#
# Failures are simulated with a stub `git` on PATH rather than by breaking a
# real repository, so the arms are deterministic and the suite needs no network,
# no remotes, and no writable checkout.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GUARD="$HERE/squad_promote_guard.sh"
WORK="$HERE/../.squad-promote-guard-test"

pass=0
fail=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

ok() { pass=$((pass + 1)); }
bad() {
  fail=$((fail + 1))
  echo "  FAIL: $*"
}

check() { # description expected_rc actual_rc
  local what="$1" want="$2" got="$3"
  if [ "$want" = "$got" ]; then
    ok
  else
    bad "$what: want rc=$want got rc=$got"
  fi
}

check_output() { # description needle haystack present|absent
  local what="$1" needle="$2" hay="$3" mode="$4"
  case "$mode" in
    present) if printf '%s' "$hay" | grep -qF -- "$needle"; then ok; else bad "$what: expected to see '$needle'"; fi ;;
    absent)  if printf '%s' "$hay" | grep -qF -- "$needle"; then bad "$what: did NOT expect '$needle'"; else ok; fi ;;
  esac
}

# ---------------------------------------------------------------------------
# A stub `git` whose behaviour is driven by files in $WORK, so each arm can make
# exactly one subcommand fail and leave the rest working.
# ---------------------------------------------------------------------------
make_stub_git() {
  mkdir -p "$WORK/bin"
  cat > "$WORK/bin/git" <<'STUB'
#!/usr/bin/env bash
# Fails whichever subcommand is named in $STUB_FAIL (space separated).
sub="${1-}"
for f in ${STUB_FAIL-}; do
  if [ "$f" = "$sub" ]; then
    # Empty stdout plus a non-zero exit: the shape a real failure takes, and
    # the shape that used to be read as "nothing found".
    echo "stub: simulated '$sub' failure" >&2
    exit 128
  fi
done
case "$sub" in
  ls-files) cat "${STUB_TRACKED:-/dev/null}" ;;
  diff)     cat "${STUB_TRACKED:-/dev/null}" ;;
  merge)    exit 0 ;;
  rm)       exit 0 ;;
  *)        exit 0 ;;
esac
STUB
  chmod +x "$WORK/bin/git"
}

# Every pre-fix arm below runs the old shape inline in its own
# `bash -euo pipefail -c`, rather than via a helper here, so what is under test
# is the literal text that was in the workflow.

echo "squad_promote_guard.sh self-test"
rm -rf "$WORK"
mkdir -p "$WORK"
make_stub_git
export PATH="$WORK/bin:$PATH"

CLEAN="$WORK/clean.txt"
DIRTY="$WORK/dirty.txt"
cat > "$CLEAN" <<'EOF'
README.md
crates/onnx-runtime-ep-cpu/src/lib.rs
docs/benchmarks/README.md
EOF
cat > "$DIRTY" <<'EOF'
README.md
.squad/team.md
crates/onnx-runtime-ep-cpu/src/lib.rs
EOF

# --- 1. the instrument's own control ---------------------------------------
out=$(STUB_FAIL="" "$GUARD" self-check 2>&1); rc=$?
check "self-check passes on the shipped list" 0 "$rc"

re=$(STUB_FAIL="" "$GUARD" forbidden-regex 2>/dev/null)
check_output "regex covers .squad/" ".squad/" "$re" present
check_output "regex covers docs/proposals/" "docs/proposals/" "$re" present

# --- 2. detection: a forbidden file must be caught (positive control) -------
out=$(STUB_FAIL="" STUB_TRACKED="$DIRTY" "$GUARD" verify-clean 2>&1); rc=$?
check "fixed: forbidden file is caught" 1 "$rc"
check_output "fixed: names the offending path" ".squad/team.md" "$out" present

out=$(STUB_TRACKED="$DIRTY" bash -euo pipefail -c '
  FORBIDDEN=$(git ls-files | grep -E "^(\.(ai-team|squad|ai-team-templates)/|team-docs/|docs/proposals/)" || true)
  if [ -n "$FORBIDDEN" ]; then echo "found"; exit 1; fi
  echo "✅ No forbidden files on preview"' 2>&1); rc=$?
check "old form also catches a real offender (so arm 3 is the only difference)" 1 "$rc"

# --- 3. THE arm: the producer fails --------------------------------------
# This is the whole point. Both forms see an empty match set; only one of them
# can tell that the emptiness came from a broken `git ls-files`.
out=$(STUB_FAIL="ls-files" STUB_TRACKED="$CLEAN" "$GUARD" verify-clean 2>&1); rc=$?
check "fixed: refuses when git ls-files fails" 1 "$rc"
check_output "fixed: says why" "indistinguishable" "$out" present
check_output "fixed: does not claim a clean tree" "No forbidden files." "$out" absent

out=$(STUB_FAIL="ls-files" STUB_TRACKED="$CLEAN" bash -euo pipefail -c '
  FORBIDDEN=$(git ls-files | grep -E "^(\.(ai-team|squad|ai-team-templates)/|team-docs/|docs/proposals/)" || true)
  if [ -n "$FORBIDDEN" ]; then echo "found"; exit 1; fi
  echo "✅ No forbidden files on preview"' 2>&1); rc=$?
check "old form reports success when git ls-files fails (the defect)" 0 "$rc"
check_output "old form prints the all-clear" "✅ No forbidden files on preview" "$out" present

# --- 4. healthy path still passes -----------------------------------------
out=$(STUB_FAIL="" STUB_TRACKED="$CLEAN" "$GUARD" verify-clean 2>&1); rc=$?
check "fixed: clean tree passes" 0 "$rc"
check_output "fixed: says so" "No forbidden files." "$out" present

# --- 5. the strip must not swallow a real failure -------------------------
out=$(STUB_FAIL="rm" "$GUARD" strip 2>&1); rc=$?
check "fixed: a failing strip is fatal" 128 "$rc"

out=$(STUB_FAIL="rm" bash -euo pipefail -c '
  git rm -rf --cached --ignore-unmatch .ai-team/ .squad/ team-docs/ || true' 2>&1); rc=$?
check "old form swallows a failing strip (the defect)" 0 "$rc"

out=$(STUB_FAIL="" "$GUARD" strip 2>&1); rc=$?
check "fixed: a working strip succeeds" 0 "$rc"

# --- 6. the merge must not fall through -----------------------------------
out=$(STUB_FAIL="merge" "$GUARD" merge-dev 2>&1); rc=$?
check "fixed: a failed merge stops the promotion" 1 "$rc"
check_output "fixed: explains the consequence" "conflict markers" "$out" present

out=$(STUB_FAIL="merge" bash -euo pipefail -c '
  git merge origin/dev --no-commit --no-ff -X theirs || true' 2>&1); rc=$?
check "old form continues past a failed merge (the defect)" 0 "$rc"

out=$(STUB_FAIL="" "$GUARD" merge-dev 2>&1); rc=$?
check "fixed: a clean merge proceeds" 0 "$rc"

# --- 7. the dry-run listing must not report a failed diff as "(none)" ------
out=$(STUB_FAIL="diff" "$GUARD" strip-list "a..b" 2>&1); rc=$?
check "fixed: a failed diff is not reported as '(none)'" 1 "$rc"
check_output "fixed: does not print (none)" "(none)" "$out" absent

out=$(STUB_FAIL="" STUB_TRACKED="$CLEAN" "$GUARD" strip-list "a..b" 2>&1); rc=$?
check "fixed: a clean range prints (none)" 0 "$rc"
check_output "fixed: prints (none)" "(none)" "$out" present

out=$(STUB_FAIL="" STUB_TRACKED="$DIRTY" "$GUARD" strip-list "a..b" 2>&1); rc=$?
check "fixed: a dirty range lists the paths" 0 "$rc"
check_output "fixed: lists the offender" ".squad/team.md" "$out" present

# --- 8. the self-check must be able to fail -------------------------------
# Otherwise every "no forbidden files" verdict above rests on an assertion that
# cannot fire. Two mutants, because the self-check makes two claims.
#
# Note what it deliberately does NOT claim: that the list is *complete*. Deleting
# an entry leaves the list and its derived regex perfectly consistent, so this
# check stays green while detection quietly narrows. Consistency is checkable
# here; completeness is a review question.

# 8a. the regex must actually match the list it was derived from.
mutant="$WORK/mutant_a.sh"
sed "s|printf '\^(%s)' \"\$out\"|printf '^(XX%s)' \"\$out\"|" "$GUARD" > "$mutant"
if ! grep -qF "'^(XX%s)'" "$mutant"; then
  bad "arm 8a anchor missed -- the mutation did not apply, so it proves nothing"
else
  out=$(STUB_FAIL="" STUB_TRACKED="$CLEAN" bash "$mutant" verify-clean 2>&1); rc=$?
  check "8a: a regex that misses its own list fails the self-check" 1 "$rc"
  check_output "8a: says which assertion broke" "self-check failed" "$out" present
fi

# 8b. and must not match everything, which would "pass" 8a trivially.
mutant="$WORK/mutant_b.sh"
# `.*` must be its own alternative: prepending it inside the group binds it to
# the first branch only, which changes nothing and would make this arm vacuous.
sed "s#printf '\^(%s)' \"\$out\"#printf '^(%s|.*)' \"\$out\"#" "$GUARD" > "$mutant"
if ! grep -qF "'^(%s|.*)'" "$mutant"; then
  bad "arm 8b anchor missed -- the mutation did not apply, so it proves nothing"
else
  out=$(STUB_FAIL="" STUB_TRACKED="$CLEAN" bash "$mutant" verify-clean 2>&1); rc=$?
  check "8b: a regex that matches ordinary sources fails the self-check" 1 "$rc"
  check_output "8b: names the over-match" "ordinary source path" "$out" present
fi

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
