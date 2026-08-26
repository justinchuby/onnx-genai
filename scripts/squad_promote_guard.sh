#!/usr/bin/env bash
#
# Guards for the squad promotion pipeline (.github/workflows/squad-promote.yml).
#
# That workflow has `contents: write`, merges dev -> preview, strips the
# team-only paths, and then merges preview -> main. Three of its steps ended in
# `|| true`, and in each one the swallowed-failure value was *the same value the
# consumer reads as success*:
#
#   git merge ... || true          a failed or conflicted merge fell through to
#                                  the strip-and-commit below, which pushes
#                                  whatever the failed merge left staged.
#   git rm --cached ... || true    the strip itself. `--ignore-unmatch` already
#                                  covers "path not present", so `|| true` could
#                                  only ever mask a real failure -- of the one
#                                  step whose whole job is keeping .squad/ off
#                                  the release branch.
#   FORBIDDEN=$(git ls-files | grep ... || true)
#                                  the backstop for the above. If `git ls-files`
#                                  fails, the pipeline yields empty, `-n` is
#                                  false, and it prints "no forbidden files" and
#                                  promotes to main. (GitHub's default `bash -e`
#                                  has no `pipefail`, so the producer's failure
#                                  is not even visible to `-e` here.)
#
# The strip and its backstop share a failure direction, so the backstop cannot
# catch a broken strip: both go quiet. Seven scratch-file-reached-main incidents
# are on record in this repo; this is a path by which an eighth arrives with a
# green check.
#
# The rule this file applies -- and the one worth carrying elsewhere -- is not
# "do not use `|| true`". Three of the tree's other uses are correct. It is:
#
#   a guard may only swallow a failure when the fallback value is
#   distinguishable, at the consumer, from the value that means success.
#
# So the producer is checked separately from the matcher. Once `git ls-files` is
# known to have run, an empty match set really does mean "found nothing", and
# `|| true` on the matcher alone is sound.
#
# Tested by scripts/squad_promote_guard_test.sh, which runs each subcommand
# against a stub `git` that can be made to fail, and asserts the pre-fix form
# passes the same arms the fixed form fails.

set -uo pipefail

# Single source of truth. The workflow previously carried the path list and the
# matching regex in two places that had already drifted (the dry-run preview
# matched `^\.(ai-team|squad|...)` with no trailing slash, the release gate
# matched `^(\.(ai-team|squad|...)/`), so a top-level file named `.squadrc`
# was forbidden by one and allowed by the other.
FORBIDDEN_PATHS=(
  ".ai-team/"
  ".squad/"
  ".ai-team-templates/"
  "team-docs/"
  "docs/proposals/"
)

forbidden_regex() {
  local p out=""
  for p in "${FORBIDDEN_PATHS[@]}"; do
    # Escape regex metacharacters; only `.` occurs today, but the list is edited
    # by hand and a future `docs/proposals+drafts/` should not become a quantifier.
    # The single quotes are deliberate: the bracket expression is a regex
    # character class, not a shell expansion.
    # shellcheck disable=SC2016
    p=$(printf '%s' "$p" | sed 's/[.[\*^$()+?{|]/\\&/g')
    out="${out:+$out|}$p"
  done
  printf '^(%s)' "$out"
}

# A regex derived from a list is only trustworthy if it actually matches the
# list. This is cheap, and it is the positive control for every use below: if it
# ever fails, every "no forbidden files" verdict in this file is meaningless.
self_check() {
  local re p rc=0
  re=$(forbidden_regex)
  for p in "${FORBIDDEN_PATHS[@]}"; do
    if ! printf '%s\n' "${p}probe.txt" | grep -qE "$re"; then
      echo "::error::forbidden-path self-check failed: '$re' does not match '${p}probe.txt'." >&2
      rc=1
    fi
  done
  # And must not match something plainly allowed, or it would match everything.
  if printf '%s\n' "crates/onnx-runtime-ep-cpu/src/lib.rs" | grep -qE "$re"; then
    echo "::error::forbidden-path self-check failed: '$re' matches an ordinary source path." >&2
    rc=1
  fi
  return "$rc"
}

merge_dev_into_preview() {
  if ! git merge origin/dev --no-commit --no-ff -X theirs; then
    echo "::error::Merging origin/dev into preview failed."
    echo "::error::Refusing to continue: the strip-and-commit that follows would"
    echo "::error::push whatever state the failed merge left staged, including"
    echo "::error::conflict markers, and the commit would look routine."
    # This `|| true` is sound: the consumer is the `exit 1` below, which does not
    # vary with the abort's result. Nothing downstream reads it.
    git merge --abort || true
    return 1
  fi
}

strip_forbidden_paths() {
  self_check || return 1
  # No `|| true`: `--ignore-unmatch` already makes "path not present" a success,
  # so any non-zero exit left here is a real failure of the strip itself.
  git rm -rf --cached --ignore-unmatch "${FORBIDDEN_PATHS[@]}"
}

verify_no_forbidden_files() {
  self_check || return 1

  local tracked re found
  # Producer checked on its own, before anything reads its output.
  if ! tracked=$(git ls-files); then
    echo "::error::Could not list tracked files. Refusing to promote:"
    echo "::error::an empty file listing is indistinguishable from 'no forbidden"
    echo "::error::files found', and this check is the last thing standing between"
    echo "::error::team-only paths and the release branch."
    return 1
  fi

  re=$(forbidden_regex)
  # Now that the producer is known to have run, `|| true` here means exactly
  # "the matcher found nothing", which is the verdict we want.
  found=$(printf '%s\n' "$tracked" | grep -E "$re" || true)

  if [ -n "$found" ]; then
    echo "::error::Forbidden files found:"
    printf '%s\n' "$found" | sed 's/^/::error::  /'
    return 1
  fi

  echo "No forbidden files."
}

preview_strip_list() {
  self_check || return 1

  local range="$1" changed re found
  if ! changed=$(git diff "$range" --name-only); then
    echo "::error::Could not diff $range. Not reporting an empty list as 'none'."
    return 1
  fi
  re=$(forbidden_regex)
  found=$(printf '%s\n' "$changed" | grep -E "$re" || true)
  if [ -z "$found" ]; then
    echo "(none)"
  else
    printf '%s\n' "$found"
  fi
}

main() {
  local cmd="${1-}"
  shift || true
  case "$cmd" in
    self-check) self_check ;;
    forbidden-regex) forbidden_regex; echo ;;
    merge-dev) merge_dev_into_preview ;;
    strip) strip_forbidden_paths ;;
    verify-clean) verify_no_forbidden_files ;;
    strip-list) preview_strip_list "${1-origin/preview..origin/dev}" ;;
    *)
      echo "usage: $0 {self-check|forbidden-regex|merge-dev|strip|verify-clean|strip-list [range]}" >&2
      return 2
      ;;
  esac
}

# Only run when executed, so the test can source it and call functions directly.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
