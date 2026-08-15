#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! rustup component list --installed | grep -q '^llvm-tools'; then
  rustup component add llvm-tools-preview
fi

if ! cargo llvm-cov --version >/dev/null 2>&1; then
  cargo install cargo-llvm-cov --locked
fi

# The final TOTAL row is workspace coverage. Rows above it are per source file;
# use --json --output-path target/llvm-cov/coverage.json for custom per-crate
# aggregation or --html --open for annotated uncovered lines.
report_args=(--summary-only)
for arg in "$@"; do
  case "$arg" in
    --json | --lcov | --cobertura | --codecov | --text | --html | --open)
      report_args=()
      break
      ;;
  esac
done

cargo llvm-cov --workspace --locked "${report_args[@]}" "$@"
