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

# THE FLOORS ARE ENV-OVERRIDABLE SO THIS SCRIPT CAN BE TESTED, AND THE DEFAULTS
# ARE THE SHIPPING CLAIM. `run-tests-guards.test.js` drives every check below
# against throwaway repositories holding four-test suites; without a seam, the
# floors alone would fail those fixtures and mask whichever guard was under test
# -- a fixture that fails for the wrong reason proves nothing, which is the
# error this suite exists to catch.
#
# This is the `fetchImpl` seam again: injectable for a test, unchanged in
# production. An override is only reachable by someone who typed it, and the
# banner prints the values in force, so a lowered floor cannot pass unnoticed.
MIN_TESTS=${MIN_TESTS:-500}
MIN_FILES=${MIN_FILES:-40}

# Local iteration writes test files before committing them. Verification runs
# must not. Default is the shipping claim; the escape hatch is loud on purpose.
allow_untracked=0
[[ ${1:-} == --allow-untracked ]] && allow_untracked=1

# Every measurement prints the container it was taken in. A relative path is a
# stale citation in space, exactly as a line number is a stale citation in time.
#
# PORCELAIN IS ON THIS BANNER DELIBERATELY AND IT IS NOT DECORATION. The same
# SHA, the same Node and the same minute produced 567 tests / 3 failures in the
# shared tree and 566 / 0 in a clean detached worktree -- the ONLY variable was
# whether the working tree was dirty. A count without its tree state is not a
# measurement, it is an anecdote. The untracked-file check below covers dirty
# TEST files; a modified tracked SOURCE file changes results and that check
# cannot see it. This line can.
echo "pwd:    $(pwd)"
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not a git tree')"
echo "head:   $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git tree')"
echo "dirty:  $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted file(s) in this tree"
echo "node:   $(node --version)"
echo "floors: ${MIN_TESTS} tests / ${MIN_FILES} files (defaults 500/40; an override is printed here so a lowered floor cannot pass unnoticed)"

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

# PROVENANCE OF THE COUNT ITSELF.
#
# `find` reads the disk. A clean clone has only what HEAD tracks. Those are
# different sets, and the difference is invisible in a total: an untracked
# test file inflates the suite by tests that nobody else can run, and the
# number is perfectly reproducible on this desk and nowhere else.
#
# We reconcile as a SET DIFFERENCE and print the offending paths. Two equal
# counts are compatible with one untracked file and one deleted file
# cancelling out, so a count comparison would certify exactly the state it
# exists to catch.
untracked=()
absent=()
provenance="unavailable (not a git tree)"
if git rev-parse --git-dir >/dev/null 2>&1; then
  head_list=$(git ls-tree -r HEAD --name-only -- . 2>/dev/null | grep '\.test\.js$' | sort)
  disk_list=$(printf '%s\n' "${test_files[@]}" | sed 's|^\./||' | sort)
  while IFS= read -r f; do [[ -n $f ]] && untracked+=("$f"); done \
    < <(comm -23 <(echo "$disk_list") <(echo "$head_list"))
  while IFS= read -r f; do [[ -n $f ]] && absent+=("$f"); done \
    < <(comm -13 <(echo "$disk_list") <(echo "$head_list"))
  provenance="${#untracked[@]} untracked, ${#absent[@]} tracked-but-missing"
fi

# IS THIS CHECKOUT EVEN COMPLETE? -- a question `porcelain 0` DOES NOT ANSWER.
#
# @e00032a4 measured a `git worktree add` that failed 14 times on a full disk,
# could not create its own `scripts/lib` directory, and left behind a directory
# that still looked like a repository. A citation harness then ran inside it and
# printed a confident GREEN, because the files it happened to read were among
# the ones that HAD been written. A half-written checkout is not a clean tree
# and it is not a dirty tree -- it is a tree that answers questions about files
# that are not there by not being asked about them.
#
# Note the two failure signatures are OPPOSITE and both are silent:
#   missing a file a test IMPORTS   -> loud, node dies, we were never at risk
#   missing a file nothing imports  -> the suite is green over a partial tree
#
# Scoped to the WHOLE REPOSITORY from the toplevel, not to this directory:
# @73e77d95 proved `git` pathspecs are silently intersected with your cwd, and
# @e00032a4's partial tree failed on `scripts/`, which a demo-scoped check would
# never have looked at. Cheap: one stat per tracked file.
incomplete=()
if git rev-parse --git-dir >/dev/null 2>&1; then
  toplevel=$(git rev-parse --show-toplevel)
  while IFS= read -r f; do
    [[ -n $f && ! -e "${toplevel}/${f}" ]] && incomplete+=("$f")
  done < <(git -C "${toplevel}" ls-tree -r HEAD --name-only 2>/dev/null)
fi

# FAIL FAST, BEFORE THE TESTS, NOT AFTER THEM.
#
# This abort was originally reported alongside the other reconciliation lines,
# AFTER the suite ran. Proved wrong by running this script inside a real
# half-written checkout (1999 of 2155 tracked files absent): the suite ran to
# completion, produced 1100 lines of output and FOUR failing tests with ENOENT
# stacks pointing at crates/, and only then said the checkout was incomplete.
# The text even read "every result below is a lie" while the results were
# ABOVE it.
#
# Four misleading reds cost more than one true one. Every test in this suite
# resolves claims against files on disk, so in an incomplete tree their
# failures describe the CHECKOUT, not the code -- and whoever reads them goes
# hunting through crates/ for a defect that is not there. Refuse to produce a
# number rather than produce one that has to be retracted.
if (( ${#incomplete[@]} > 0 )); then
  echo "FAIL: this checkout is INCOMPLETE -- ${#incomplete[@]} of $(git -C "${toplevel}" ls-tree -r HEAD --name-only | wc -l | tr -d ' ') file(s) tracked at HEAD are absent." >&2
  echo "      This is not a dirty tree and it is not a stale one; it is a" >&2
  echo "      half-written checkout, and a green from it would be a spotless" >&2
  echo "      measurement of a repository that was never assembled." >&2
  echo "      NO TESTS WERE RUN. Fix the checkout, then re-run." >&2
  echo "      A full disk produces exactly this: 'git worktree add' fails" >&2
  echo "      partway and still leaves a directory that looks like a repo." >&2
  printf '      %s\n' "${incomplete[@]:0:10}" >&2
  (( ${#incomplete[@]} > 10 )) && echo "      ... and $(( ${#incomplete[@]} - 10 )) more" >&2
  exit 1
fi

# Same basename in two directories: legal, and tonight it shipped twice with
# DIFFERENT BYTES while one documented command reached only one of them.
dupes=$(printf '%s\n' "${test_files[@]}" | sed 's|.*/||' | sort | uniq -d)

output=$(node --test "${test_files[@]}" 2>&1)
node_status=$?
echo "$output"

field() { echo "$output" | grep -E "^. $1 " | tail -1 | awk '{print $3}'; }

tests=$(field tests)
failed=$(field fail)
suites=$(field suites)

echo ""
echo "── reconciliation ─────────────────────────────"
echo "  tree             : $(pwd)"
echo "  head / dirty     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git') / $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted"
echo "  discovered files : ${discovered}"
echo "  suites executed  : ${suites:-<unparsed>}"
echo "  tests            : ${tests:-<unparsed>}"
echo "  failed           : ${failed:-<unparsed>}"
echo "  provenance       : ${provenance}"
echo "  checkout         : ${#incomplete[@]} tracked file(s) missing from this working tree"

status=0

if [[ -n ${dupes} ]]; then
  echo "WARN: the same test filename appears in more than one directory:" >&2
  while IFS= read -r d; do
    [[ -z $d ]] && continue
    echo "      ${d}" >&2
    printf '%s\n' "${test_files[@]}" | grep -- "/${d}$" | sed 's/^/        /' >&2
  done <<< "${dupes}"
  echo "      Not fatal. But a glob that reaches one copy and not the other" >&2
  echo "      reports a stable total whose meaning silently differs." >&2
fi



if (( ${#absent[@]} > 0 )); then
  # THIS IS THE CHECK THAT CATCHES A NARROWED DISCOVERY, AND IT IS THE MOST
  # IMPORTANT ONE IN THIS FILE. I first recorded it as dead code, because a
  # DELETED tracked test file is caught earlier by the incomplete-checkout
  # abort. That was a true observation and the wrong conclusion, and the
  # mutation that was supposed to confirm it refuted it instead:
  #
  #   narrow the `find` above to `-maxdepth 1`, at 9b54d3a9, in a detached
  #   worktree -> `nested/beta.test.js` is TRACKED and PRESENT ON DISK, so the
  #   incomplete-checkout abort correctly stays silent, and THIS branch is the
  #   only thing standing between a silently smaller run and a green.
  #
  # So the two checks are not duplicates. The abort answers "is the tree all
  # here"; this answers "did I look at all of it". A file can be present and
  # unseen, which is the exact failure mode -- a documented glob skipping 305
  # tests and exiting 0 -- that this whole script was written for.
  echo "FAIL: ${#absent[@]} test file(s) are tracked at HEAD but were NOT RUN:" >&2
  printf '      %s\n' "${absent[@]}" >&2
  echo "      They are tracked at HEAD and this run did not execute them." >&2
  echo "      Either they are missing from disk (a broken checkout), or they" >&2
  echo "      are on disk and discovery did not reach them (a narrowed glob)." >&2
  echo "      Check which: a present-but-unseen file is the failure this" >&2
  echo "      script exists to catch, and it exits 0 everywhere else." >&2
  status=1
fi

if (( ${#untracked[@]} > 0 )); then
  if (( allow_untracked )); then
    echo "WARN: ${#untracked[@]} untracked test file(s) were INCLUDED in this count:" >&2
    printf '      %s\n' "${untracked[@]}" >&2
    echo "      This total describes this working tree, NOT the branch. Do not" >&2
    echo "      quote it as a property of the shipping tree." >&2
  else
    echo "FAIL: ${#untracked[@]} test file(s) ran here but are not committed:" >&2
    printf '      %s\n' "${untracked[@]}" >&2
    echo "      A clean clone does not have them, so this total is a claim about" >&2
    echo "      this desk rather than about the branch. Commit them, or re-run" >&2
    echo "      with --allow-untracked to accept a desk-scoped number knowingly." >&2
    status=1
  fi
fi

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
  # RE-EMIT THE DIAGNOSIS LAST, ON PURPOSE.
  #
  # Node prints its `✖ failing tests:` section BEFORE this reconciliation block,
  # so `run-tests.sh | tail -12` -- the obvious thing an operator types -- shows
  # the word FAIL and destroys the only copy of WHICH test failed. That is not
  # hypothetical: two intermittent reds were observed and lost exactly this way,
  # and eighteen subsequent runs were green, so the failing test could never be
  # named. AN UNIDENTIFIABLE RED IS WORSE THAN A GREEN; it makes every later run
  # unfalsifiable evidence.
  #
  # The runner already holds the whole run in `$output`. Printing the section
  # again costs nothing and makes the diagnosis survive any pipeline.
  detail=$(echo "$output" | awk '/^✖ failing tests:/{f=1} f')
  if [[ -n ${detail} ]]; then
    echo "      --- failing tests, re-printed so a piped run keeps them ---" >&2
    echo "${detail}" | sed 's/^/      /' >&2
    # And the NAMES again, last. The detail block above is ~20 lines per
    # failure, so `| tail` lands inside a stack trace and still cannot tell you
    # what broke. The last line of this script's output must be the answer.
    names=$(echo "${detail}" | grep '✖' | grep -v '^✖ failing tests:')
    if [[ -n ${names} ]]; then
      echo "      --- the ${failed} failing test(s), by name ---" >&2
      echo "${names}" | sed 's/^ *//; s/^/      /' >&2
    fi
  else
    echo "      Node reported ${failed} failure(s) but this script could not" >&2
    echo "      locate its failure section to re-print. Re-run WITHOUT piping" >&2
    echo "      and read Node's own output; do not treat this as diagnosed." >&2
  fi
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
