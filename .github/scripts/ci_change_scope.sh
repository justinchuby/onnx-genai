#!/usr/bin/env bash

# Classify one Git path for the CI docs-only optimization.
#
# The optional second argument is the newline-delimited set of files embedded
# into Rust sources. Return 0 only when the path is safe to treat as docs-only.
#
# Extension-normalization contract (also consumed by the Rust fixture census):
# 1. normalize `\` to `/` and retain that exact string for equality/diagnostics;
# 2. on a separate classification copy, strip trailing ASCII whitespace only:
#    space, tab, CR, LF, vertical tab, and form feed;
# 3. fail closed if stripping leaves an empty path or empty terminal filename;
# 4. ASCII-lowercase that copy, then inspect only its terminal extension.
# Interior whitespace and trailing dots are not normalized.
ci_is_docs_path() {
  local path="${1//\\//}"
  local extension_path
  local embedded="${2-}"
  local target

  # Git paths are case-sensitive data, so keep `path` unchanged for embedded
  # source equality. Strip only the six documented trailing ASCII-whitespace
  # bytes from the extension-classification copy.
  extension_path="$path"
  while [[ -n "$extension_path" ]]; do
    case "${extension_path: -1}" in
      ' '|$'\t'|$'\r'|$'\n'|$'\v'|$'\f')
        extension_path="${extension_path::-1}"
        ;;
      *)
        break
        ;;
    esac
  done
  if [[ -z "$extension_path" || "$extension_path" == */ ]]; then
    return 1
  fi

  # Extension policy is case-insensitive: ONNX tooling treats `.onnx`,
  # `.ONNX`, and mixed-case spellings as the same graph format.
  extension_path="$(printf '%s' "$extension_path" | LC_ALL=C tr '[:upper:]' '[:lower:]')"

  # Graph fixtures are executable test/source inputs even when somebody puts
  # them below docs/. This precedence is the invariant: generic directory
  # classification must never hide the binary/textproto census. Textproto
  # casing is allowed by the fixture policy, but every casing still forces the
  # Fast census; binary ONNX casing is rejected by that census.
  case "$extension_path" in
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
  local workflow=".github/workflows/ci.yml"

  local graph_path
  for graph_path in \
    "docs/foo/model.onnx" \
    "docs/foo/model.ONNX" \
    "docs/foo/model.OnNx" \
    "docs/foo/model.onnx.textproto" \
    "docs/foo/model.ONNX.TEXTPROTO" \
    "docs/foo/model.OnNx.TeXtPrOtO" \
    'docs\foo\model.onnx' \
    'docs\foo\model.ONNX' \
    'docs\foo\model.OnNx' \
    'docs\foo\model.onnx.textproto' \
    'docs\foo\model.ONNX.TEXTPROTO' \
    'docs\foo\model.OnNx.TeXtPrOtO'
  do
    if ci_is_docs_path "$graph_path"; then
      echo "change-scope self-test: $graph_path was misclassified as docs-only" >&2
      failures=$((failures + 1))
    fi
  done
  local whitespace
  for whitespace in ' ' $'\t' $'\r' $'\n' $'\v' $'\f'; do
    for graph_path in \
      "docs/foo/model.ONNX${whitespace}" \
      "docs/foo/model.OnNx.TeXtPrOtO${whitespace}"
    do
      if ci_is_docs_path "$graph_path"; then
        echo "change-scope self-test: trailing ASCII whitespace bypassed graph policy: $(printf %q "$graph_path")" >&2
        failures=$((failures + 1))
      fi
    done
  done
  if ! ci_is_docs_path "docs/foo/model.ONNX.md"; then
    echo "change-scope self-test: terminal .md double extension stopped being docs-only" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "" || ci_is_docs_path "   " || ci_is_docs_path $'docs/foo/\t'; then
    echo "change-scope self-test: empty/all-whitespace terminal filename did not fail closed" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "docs/foo/model .ONNX"; then
    echo "change-scope self-test: interior whitespace was incorrectly removed" >&2
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
  if ci_is_docs_path "docs/compiled.md " "docs/compiled.md "; then
    echo "change-scope self-test: trailing whitespace changed exact embedded equality" >&2
    failures=$((failures + 1))
  fi
  if ! ci_is_docs_path "docs/compiled.md " "docs/compiled.md"; then
    echo "change-scope self-test: extension trimming leaked into embedded equality" >&2
    failures=$((failures + 1))
  fi
  if ! ci_is_docs_path "docs/Compiled.md" "docs/compiled.md"; then
    echo "change-scope self-test: embedded equality became case-insensitive" >&2
    failures=$((failures + 1))
  fi
  if ci_is_docs_path "src/lib.rs"; then
    echo "change-scope self-test: Rust source was misclassified as docs-only" >&2
    failures=$((failures + 1))
  fi
  if ! grep -Eq '^[[:space:]]*&&[[:space:]]+\.[[:space:]]+\.github/scripts/ci_change_scope\.sh' "$workflow" \
    || ! grep -Eq '^[[:space:]]*if ! ci_is_docs_path "\$f" "\$embedded"; then' "$workflow"; then
    echo "change-scope self-test: ci.yml is not sourcing and calling this classifier" >&2
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
