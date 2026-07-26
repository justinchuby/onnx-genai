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

run_tlc() {
    local module="$1"
    local config="$2"
    "${java_bin}" -XX:+UseParallelGC -jar "${TLA2TOOLS_JAR}" \
        -workers "${workers}" \
        -metadir "${meta_root}/${config}" \
        -config "${spec_dir}/${config}.cfg" \
        "${spec_dir}/${module}.tla"
}

run_model() {
    run_tlc "$1" "${2:-$1}"
}

# Negative model: the configuration must violate the named invariant. A model
# that stops catching its own counterexample is a silently weakened model.
expect_violation() {
    local module="$1"
    local config="$2"
    local invariant="$3"
    local output
    if output="$(run_tlc "${module}" "${config}" 2>&1)"; then
        printf '%s\n' "${output}" >&2
        echo "expected ${config} to violate ${invariant}, but TLC succeeded" >&2
        return 1
    fi
    if ! printf '%s' "${output}" | grep -q "Invariant ${invariant} is violated"; then
        printf '%s\n' "${output}" >&2
        echo "expected ${config} to violate ${invariant}" >&2
        return 1
    fi
    echo "${config}: ${invariant} violated as expected"
}

run_model PressureProtocol
run_model CollectiveOrdering
run_model BufferOwnership
run_model KvAdmission
expect_violation KvAdmission KvAdmissionUnguarded ProgressPossible
run_model NodeFailure
expect_violation NodeFailure NodeFailureUnguarded FailedNodeStopsAtFault
run_model CoResidency
expect_violation CoResidency CoResidencyUnguarded NoWastedResidency
