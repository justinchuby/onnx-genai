#!/usr/bin/env bash

# Classify one Git path for the CI docs-only optimization.
#
# The optional second argument is the newline-delimited set of files embedded
# into Rust sources. Return 0 only when the path is safe to treat as docs-only.
ci_is_docs_path() {
  local path="${1//\\//}"
  local embedded="${2-}"
  local target

  # Graph fixtures are executable test/source inputs even when somebody puts
  # them below docs/. This precedence is the invariant: generic directory
  # classification must never hide the binary/textproto census.
  case "$path" in
    *.onnx|*.textproto)
      return 1
      ;;
    docs/*|wiki/*|*.md|*.mdx|*.markdown|LICENSE|NOTICE)
      ;;
    *)
      return 1
      ;;
  esac

  while IFS= read -r target; do
    target="${target//\\//}"
    if [[ -n "$target" && "$target" == "$path" ]]; then
      return 1
    fi
  done <<< "$embedded"
  return 0
}

ci_change_scope_self_test() {
  local failures=0

  if ci_is_docs_path "docs/example/model.onnx"; then
    echo "change-scope self-test: docs/example/model.onnx was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "docs/example/model.onnx.textproto"; then
    echo "change-scope self-test: docs/example/model.onnx.textproto was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path 'docs\example\model.onnx'; then
    echo "change-scope self-test: Windows-form binary ONNX path was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path 'docs\example\model.onnx.textproto'; then
    echo "change-scope self-test: Windows-form textproto path was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ! ci_is_docs_path "docs/example/readme.md"; then
    echo "change-scope self-test: ordinary docs markdown stopped being docs-only" >&2
    failures=$((failures + 1))
  fi
  if ! ci_is_docs_path 'docs\example\readme.md'; then
    echo "change-scope self-test: Windows-form docs markdown stopped being docs-only" >&2
    failures=$((failures + 1))
  fi
  if ! ci_is_docs_path "wiki/design.mdx"; then
    echo "change-scope self-test: ordinary wiki content stopped being docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "docs/compiled.md" "docs/compiled.md"; then
    echo "change-scope self-test: embedded markdown was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "src/lib.rs"; then
    echo "change-scope self-test: Rust source was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi

  if [[ "$failures" -ne 0 ]]; then
    return 1
  fi
  echo "change-scope self-test: graph fixtures force code CI; ordinary docs remain docs-only"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "${1-}" != "--self-test" || "$#" -ne 1 ]]; then
    echo "usage: $0 --self-test" >&2
    exit 2
  fi
  ci_change_scope_self_test
fi
