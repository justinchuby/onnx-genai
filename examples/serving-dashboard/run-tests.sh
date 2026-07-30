#!/usr/bin/env bash
#
# The one way to run this suite.
#
# Seven tracked files documented this command in five incompatible forms, and
# every one of them was a hand-maintained copy of a fact. This script IS the
# fact. Cite it; do not restate it.
#
# THREE THINGS THIS EXISTS TO PREVENT, ALL THREE OBSERVED IN THIS REPOSITORY:
#
#   1. `node --test 'glob'` DOES NOT RECURSE. Recursion is a property of being
#      given no arguments, not a property of the Node version. A documented
#      glob silently skipped 305 tests -- the entire honesty layer -- and
#      exited 0.
#
#   2. A HARDCODED DIRECTORY LIST STOPS COVERING WHATEVER WAS ADDED LAST, which
#      is always the thing most likely to be wrong. Four reviewers independently
#      specified `'*.test.js' 'dashboard/*.test.js'` and all four missed
#      `ui/`. So this script DISCOVERS test files instead of listing them: a new
#      directory is covered the moment it exists, with no edit here.
#
#   3. `node --test` TREATS "NO FILES MATCHED" AS SUCCESS. A runner that
#      silently executes a subset is the exact defect it exists to catch, so
#      the discovered file count is reconciled against the suite count Node
#      reports. A run that loses a file fails LOUDLY rather than passing
#      smaller.
#
# THE FLOOR BELOW IS AN ANTI-VACUITY CHECK, NOT A COMPLETENESS CHECK. A floor
# perfectly detects an empty run and is blind to the silent shrinkage of one --
# a duplicate key once walked straight through a `MIN_ENTRIES = 36` guard
# because it arrived as an addition performing a subtraction. The file-count
# reconciliation, not the floor, is what catches shrinkage.

set -uo pipefail

cd "$(dirname "$0")" || exit 1

MIN_TESTS=500
MIN_FILES=40

# Every measurement prints the container it was taken in. A relative path is a
# stale citation in space, exactly as a line number is a stale citation in time.
echo "pwd:  $(pwd)"
echo "head: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git tree')"
echo "node: $(node --version)"

# Discover, never enumerate.
test_files=()
while IFS= read -r file; do
  test_files+=("$file")
done < <(find . -name '*.test.js' -not -path './node_modules/*' | sort)

discovered=${#test_files[@]}
echo "discovered: $discovered test files"

if [[ $discovered -eq 0 ]]; then
  echo "FAIL: discovered no test files at all. The suite cannot be green because" >&2
  echo "      it was never run. This is the failure that looks like success." >&2
  exit 1
fi

output=$(node --test "${test_files[@]}" 2>&1)
node_status=$?
echo "$output"

field() { echo "$output" | grep -E "^. $1 " | tail -1 | awk '{print $3}'; }

tests=$(field tests)
failed=$(field fail)
suites=$(field suites)

echo ""
echo "── reconciliation ─────────────────────────────"
echo "  discovered files : ${discovered}"
echo "  suites executed  : ${suites:-<unparsed>}"
echo "  tests            : ${tests:-<unparsed>}"
echo "  failed           : ${failed:-<unparsed>}"

status=0

if [[ -z ${tests:-} || -z ${failed:-} || -z ${suites:-} ]]; then
  echo "FAIL: could not parse Node's summary. Refusing to report a result we" >&2
  echo "      could not read -- an unparsed run is not a passing run." >&2
  exit 1
fi

# THE CHECK THAT CATCHES SHRINKAGE.
#
# NOTE, MEASURED RATHER THAN ASSUMED: `suites` is NOT the number of files. Node
# counts `describe` blocks, so 46 files report 90 suites. The first version of
# this script asserted one-suite-per-file and failed instantly on a perfectly
# green tree -- which is the correct behaviour for a guard built on a premise
# nobody checked, and is why the assertion is stated as an inequality now.
#
# Every file is passed to Node explicitly, so "run but not discovered" is not
# reachable. What IS reachable is a file that is discovered, loads, and
# contributes nothing -- an emptied test file, or one whose suites were
# commented out. That file still counts as discovered and vanishes from the
# suite total, so the inequality is what catches it.
if [[ $suites -lt $discovered ]]; then
  echo "FAIL: found ${discovered} test files but only ${suites} suites ran." >&2
  echo "      At least one discovered file contributed no tests at all." >&2
  status=1
fi

if [[ $discovered -lt $MIN_FILES ]]; then
  echo "FAIL: ${discovered} test files is below the ${MIN_FILES} floor." >&2
  status=1
fi

if [[ $tests -lt $MIN_TESTS ]]; then
  echo "FAIL: ${tests} tests is below the ${MIN_TESTS} floor. Either the suite" >&2
  echo "      shrank or the runner stopped finding files. A check that stopped" >&2
  echo "      looking and a check that found nothing print the same green." >&2
  status=1
fi

if [[ $failed -ne 0 ]]; then
  echo "FAIL: ${failed} failing test(s)." >&2
  status=1
fi

if [[ $node_status -ne 0 && $status -eq 0 ]]; then
  echo "FAIL: node exited ${node_status} while every summary line looked clean." >&2
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "PASS: ${tests} tests across ${suites} suites, 0 failures."
fi

exit $status
