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

# THE FLOOR ABOVE IS FOR VACUITY. THE RATCHET BELOW IS FOR COMPLETENESS.
# They are different guards and neither can do the other's job.
#
#   MIN_TESTS=500 PASSES AT 595 WHEN THE TRUE COUNT IS 600.
#
# That is not a tuning mistake, it is structural: a floor low enough to be
# stable while the suite grows is BY CONSTRUCTION unable to see a one-suite
# loss. And a whole suite that silently fails to load looks precisely like a
# smaller number. This cost 588 -> 435 tonight without a single red.
#
# A DROPPING TEST COUNT IS A LOAD FAILURE WEARING A TEST FAILURE'S CLOTHES.
#
# The floor is NOT replaced by the ratchet, deliberately: on a fresh clone the
# baseline file may be absent, and a ratchet with no baseline guards nothing at
# all. The floor covers the ratchet's cold start; the ratchet covers the
# floor's slack. Deleting either one reopens a hole the other cannot see.
#
# THE SEAM IS THE BASELINE PATH, NOT A DISABLE SWITCH, AND THE FIRST VERSION OF
# THIS GOT IT WRONG IN THE EXACT WAY THIS SCRIPT EXISTS TO CATCH.
#
# That version disabled the ratchet whenever MIN_TESTS/MIN_FILES were
# overridden, reasoning that a four-test fixture repository must not trip a
# ratchet built for a 600-test suite. The reasoning was right and the mechanism
# was wrong: the self-proof harness lowers the floors on EVERY case, so the
# ratchet was disabled in all eleven of them. IT WAS A GUARD THAT COULD NOT BE
# EXECUTED BY THE SUITE THAT EXISTS TO EXECUTE IT -- green forever, over a
# corpus it never read, which is the defect this whole file was written about.
#
# Fixture safety needs no switch: a scratch repository has no baseline file, so
# the ratchet seeds itself and passes. Absence already means "nothing to compare
# against". Pointing BASELINE at a fixture path is what makes the DROP path
# reachable from a test, so the guard is now provable rather than merely
# present.
BASELINE_FILE=${TEST_COUNT_BASELINE:-test-count.baseline}

# NO BASELINE IS COMMITTED TONIGHT, AND THAT IS A DECISION RATHER THAN AN
# OMISSION. Seeding one requires a count somebody can reproduce, and HEAD is
# moving about once a minute with several agents' files mid-edit -- the number
# measured minutes ago is already a claim about a tree that no longer exists.
# Committing it would hand twelve other agents an immediate red they would
# reasonably attribute to their own work. A false red is not the safe direction
# of the same mistake; it is a different failure, and it spends the one resource
# a ship night has none of.
#
# So this is the ratchet's own rule applied to the person installing it: it
# refuses to bank a baseline from a run nobody can reproduce, and so did I. The
# mechanism lands proven, the first green run on a settled tree seeds it, and
# whoever commits that number should be able to say which tree it came from.

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
echo "ratchet: $( [[ -f $BASELINE_FILE ]] && echo "baseline $(tr -cd '0-9' < "$BASELINE_FILE") from ${BASELINE_FILE}" || echo "no baseline yet (${BASELINE_FILE} absent; will be seeded)" )"

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

head_before=$(git rev-parse HEAD 2>/dev/null || echo 'no-git')
output=$(node --test "${test_files[@]}" 2>&1)
node_status=$?
head_after=$(git rev-parse HEAD 2>/dev/null || echo 'no-git')
echo "$output"

field() { echo "$output" | grep -E "^. $1 " | tail -1 | awk '{print $3}'; }

tests=$(field tests)
failed=$(field fail)
suites=$(field suites)

echo ""
echo "── reconciliation ─────────────────────────────"
echo "  tree             : $(pwd)"
echo "  head / dirty     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git') / $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted"
echo "  head before/after: ${head_before:0:8} -> ${head_after:0:8}$([[ ${head_before} != "${head_after}" ]] && echo '  ⛔ MOVED MID-RUN')"
echo "  discovered files : ${discovered}"
echo "  suites executed  : ${suites:-<unparsed>}"
echo "  tests            : ${tests:-<unparsed>}"
echo "  failed           : ${failed:-<unparsed>}"
echo "  provenance       : ${provenance}"
echo "  checkout         : ${#incomplete[@]} tracked file(s) missing from this working tree"

status=0

# DID THE TREE HOLD STILL WHILE WE MEASURED IT?
#
# THIS SUITE TAKES MINUTES AND FOURTEEN AGENTS COMMIT TO THIS BRANCH, SO HEAD
# MOVES DURING RUNS. Many tests here resolve claims against `HEAD` rather than
# the worktree -- deliberately, so a dirty desk cannot fake a green -- which
# means a commit landing mid-run gives EARLY files one tree and LATE files
# another. The result is a red that does not reproduce.
#
# OBSERVED, NOT HYPOTHESISED: one run reported 4 failures including a CAN RUN
# control, and `prefix-counters-forbidden.test.js` then passed 4/4 in isolation
# seconds later. Nothing was wrong with it. HEAD had moved underneath the run.
#
# A phantom red is expensive in both directions: chased, it costs an hour on a
# defect that does not exist; dismissed, it trains everyone to wave away the
# real one. Neither is acceptable, and the only honest output is to say the
# measurement has no single subject.
#
# NOTE THIS IS NOT THE `dirty` COUNT. A clean porcelain at the start and a clean
# porcelain at the end are perfectly compatible with two different commits.
if [[ ${head_before} != "${head_after}" ]]; then
  echo "FAIL: HEAD MOVED WHILE THE SUITE WAS RUNNING." >&2
  echo "      before: ${head_before}" >&2
  echo "      after:  ${head_after}" >&2
  echo "      Tests here resolve claims against HEAD, so files that ran early" >&2
  echo "      and files that ran late were graded against DIFFERENT TREES." >&2
  echo "      Every count above describes no single commit. Any failure may be" >&2
  echo "      a phantom, and any pass may be stale. RE-RUN; do not triage this" >&2
  echo "      output, and do not quote its numbers." >&2
  status=1
fi

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

# THE RATCHET. Fails on ANY decrease, not just a decrease past the floor.
#
# DETECTION AND ADVANCEMENT ARE SPLIT ON PURPOSE, AND THE FIRST VERSION OF THIS
# BLOCK GOT IT WRONG. It skipped everything unless `failed -eq 0`, reasoning
# that a red run's count measures how far the run got rather than how big the
# suite is. That is true of ADVANCING and false of DETECTING, and on this tree
# the difference is everything: the shared worktree is red most of the time
# (2 pre-existing failures as of this writing), so a ratchet gated on green
# WOULD NEVER HAVE RUN AT ALL -- correct, and vacuous, which is the exact
# failure this suite exists to catch.
#
# So: a DROP is reported whenever it is seen, red or green. The baseline only
# ADVANCES on a clean run, because a high-water mark banked from a partial run
# is a number nobody can reproduce.
#
# The two facts are also different diagnoses and both matter:
#   "2 tests failed"        -> a test broke
#   "40 tests never ran"    -> a suite failed to LOAD
# The second is the one nothing else here can see.
if [[ -f $BASELINE_FILE ]]; then
  baseline=$(tr -cd '0-9' < "$BASELINE_FILE")
  if [[ -z $baseline ]]; then
    echo "FAIL: ${BASELINE_FILE} exists but holds no number. A ratchet that" >&2
    echo "      cannot read its baseline must not silently become no ratchet." >&2
    status=1
  elif [[ $tests -lt $baseline ]]; then
    # THE OVERRIDE MUST NAME THE EXACT NEW COUNT, not merely assert consent.
    # `ALLOW_TEST_COUNT_DROP=1` would be typed once, forgotten in a shell,
    # and would then wave through every later drop -- which is how a ratchet
    # decays into the floor it was meant to replace. Naming the count makes
    # the permission expire on its own the next time the number moves.
    dropped=$((baseline - tests))
    if [[ ${ALLOW_TEST_COUNT_DROP:-} == "$tests" ]]; then
      echo "note: test count dropped ${baseline} -> ${tests}, allowed by" >&2
      echo "      ALLOW_TEST_COUNT_DROP=${tests}." >&2
      [[ $failed -eq 0 ]] && echo "$tests" > "$BASELINE_FILE"
    else
      echo "FAIL: test count DROPPED ${baseline} -> ${tests} (${dropped} fewer)." >&2
      if [[ $failed -eq 0 ]]; then
        echo "      NO TEST FAILED, so this is not a broken test -- it is tests" >&2
        echo "      that never ran. Look for a suite that failed to LOAD before" >&2
        echo "      you look for a suite that failed." >&2
      else
        echo "      ${failed} test(s) also failed. Do not assume one explains the" >&2
        echo "      other: a suite that aborts on load removes its whole count," >&2
        echo "      and ${dropped} missing is not the same fact as ${failed} failing." >&2
      fi
      echo "      If the removal is deliberate, re-run with:" >&2
      echo "        ALLOW_TEST_COUNT_DROP=${tests} $0" >&2
      status=1
    fi
  elif [[ $tests -gt $baseline && $failed -eq 0 ]]; then
    # Advance automatically. A ratchet that only PRINTS its new high-water
    # mark is a suggestion, and the next drop is measured against whatever
    # stale number nobody got around to committing.
    echo "$tests" > "$BASELINE_FILE"
    echo "ratchet: test count ${baseline} -> ${tests}; ${BASELINE_FILE} advanced (commit it)."
  fi
elif [[ $failed -eq 0 ]]; then
  echo "$tests" > "$BASELINE_FILE"
  echo "ratchet: no baseline yet; seeded ${BASELINE_FILE} at ${tests} (commit it)."
else
  # Say so out loud. A guard that declined to arm itself must not be
  # indistinguishable from a guard that armed and was satisfied.
  echo "ratchet: NOT seeded -- ${failed} test(s) failed, and a baseline banked" >&2
  echo "         from a red run is a number nobody can reproduce." >&2
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
