#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  crates/onnx-runtime-ep-cpu/benches/run_einsum.sh census OWNER PHYSICAL_CORES ABSOLUTE_TARGET_DIR
  crates/onnx-runtime-ep-cpu/benches/run_einsum.sh run    OWNER PHYSICAL_CORES ABSOLUTE_TARGET_DIR

Run from the repository root. The target directory must be dedicated to this
evidence sweep; "run" refuses existing Einsum Criterion results.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi

if [[ $# -ne 4 ]]; then
  usage >&2
  exit 2
fi

mode=$1
owner=$2
physical_cores=$3
target_dir=$4

case "$mode" in
  census | run) ;;
  *)
    echo "invalid mode '$mode': expected 'census' or 'run'" >&2
    exit 2
    ;;
esac

if [[ ! $owner =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid owner '$owner': use only letters, digits, '_', '.', or '-'" >&2
  exit 2
fi
if [[ ! $physical_cores =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid physical-core budget '$physical_cores': expected a positive integer" >&2
  exit 2
fi
if [[ $target_dir != /* ]]; then
  echo "invalid target directory '$target_dir': CARGO_TARGET_DIR must be absolute" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"

mkdir -p "$target_dir"
export ONNX_GENAI_CPU_DECODE_THREADS="$physical_cores"
export CARGO_TARGET_DIR="$target_dir"

log="$target_dir/einsum-$mode.log"
reason="CPU Einsum $mode (26 selectors)"
args=(cargo bench -p onnx-runtime-ep-cpu --bench einsum --)
if [[ $mode == census ]]; then
  args+=(--list)
else
  args+=(--noplot)
fi

scripts/hostlock.sh run --owner "$owner" --reason "$reason" \
  --wait --gate 3 --strict-reap -- "${args[@]}" 2>&1 | tee "$log"

if [[ $mode == census ]]; then
  selector_count="$(grep -c ': benchmark$' "$log" || true)"
  if [[ $selector_count -ne 12 ]]; then
    echo "selector census failed: found $selector_count Criterion selectors, expected 12" >&2
    exit 1
  fi
  grep -Eq '^EINSUM_CENSUS_COMPLETE .* selector_count=26$' "$log" || {
    echo "selector census failed: missing EINSUM_CENSUS_COMPLETE selector_count=26" >&2
    exit 1
  }
  echo "selector census passed: 26/26"
else
  grep -Eq '^EINSUM_EVIDENCE_COMPLETE .* selector_count=26$' "$log" || {
    echo "evidence sweep failed: missing EINSUM_EVIDENCE_COMPLETE selector_count=26" >&2
    exit 1
  }
fi
