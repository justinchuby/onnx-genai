#!/usr/bin/env bash
set -euo pipefail

: "${TLA2TOOLS_JAR:?set TLA2TOOLS_JAR to a pinned tla2tools.jar}"

java_bin="${JAVA_BIN:-java}"
workers="${TLC_WORKERS:-1}"
spec_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
meta_root="$(mktemp -d "${TMPDIR:-/tmp}/onnx-genai-tlc.XXXXXX")"

cleanup() {
    rm -rf "${meta_root}"
}
trap cleanup EXIT

run_model() {
    local module="$1"
    "${java_bin}" -XX:+UseParallelGC -jar "${TLA2TOOLS_JAR}" \
        -workers "${workers}" \
        -metadir "${meta_root}/${module}" \
        -config "${spec_dir}/${module}.cfg" \
        "${spec_dir}/${module}.tla"
}

run_model PressureProtocol
run_model CollectiveOrdering
run_model BufferOwnership
