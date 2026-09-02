#!/usr/bin/env bash
# Trustworthy PR #2345 evidence entry point. The outer hostlock must cover the
# complete f16+f32 sweep, not individual arms:
#
#   CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_DEVICE=0 \
#     scripts/hostlock.sh run --owner pris --reason "PR 2345 CUDA Einsum sweep" \
#       --wait --gate 3 --strict-reap -- scripts/bench_cuda_einsum.sh

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one physical index or UUID}"
case "$CUDA_VISIBLE_DEVICES" in
    *,*|'') echo "CUDA_VISIBLE_DEVICES must contain exactly one device selector" >&2; exit 2 ;;
esac
: "${HOSTLOCK_OWNER:?run through scripts/hostlock.sh run --owner <name>}"

export ONNX_GENAI_CUDA_DEVICE="${ONNX_GENAI_CUDA_DEVICE:-0}"
export ONNX_GENAI_CUDA_PHYSICAL_DEVICE="${ONNX_GENAI_CUDA_PHYSICAL_DEVICE:-$CUDA_VISIBLE_DEVICES}"
if [[ "$ONNX_GENAI_CUDA_DEVICE" != 0 ]]; then
    echo "ONNX_GENAI_CUDA_DEVICE must be logical ordinal 0 after CUDA_VISIBLE_DEVICES remapping" >&2
    exit 2
fi
if [[ "$ONNX_GENAI_CUDA_PHYSICAL_DEVICE" != "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "ONNX_GENAI_CUDA_PHYSICAL_DEVICE must equal CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi

PROVENANCE=$(scripts/hostlock.sh provenance --oneline --expect-runnable "${EINSUM_BENCH_EXPECT_RUNNABLE:-3}")
printf 'HOSTLOCK_SCRIPT,%s\n' "$PROVENANCE"
for required in \
    hostlock_state=HELD declared=yes held_owner_source=flag \
    contended=no lock_scope=box takeover=none
do
    if [[ " $PROVENANCE " != *" $required "* ]]; then
        echo "benchmark refused: hostlock provenance lacks $required: $PROVENANCE" >&2
        exit 2
    fi
done
if [[ " $PROVENANCE " != *" gate=satisfied:"* ]]; then
    echo "benchmark refused: hostlock gate was not satisfied: $PROVENANCE" >&2
    exit 2
fi

STATUS=$(git status --porcelain)
if [[ -n "$STATUS" ]]; then
    echo "benchmark refused: worktree must be clean so commit/tree provenance is exact" >&2
    printf '%s\n' "$STATUS" >&2
    exit 2
fi

printf 'BUILD_COMMIT,%s\n' "$(git rev-parse HEAD)"
printf 'BUILD_TREE,%s\n' "$(git rev-parse 'HEAD^{tree}')"
printf 'BUILD_BRANCH,%s\n' "$(git symbolic-ref --short -q HEAD || printf detached)"
printf 'BUILD_PATH,package=onnx-runtime-ep-cuda,manifest=%s,test=crates/onnx-runtime-ep-cuda/tests/einsum_gpu.rs,features=cuda|gpu-tests\n' \
    "$ROOT/crates/onnx-runtime-ep-cuda/Cargo.toml"
cargo tree -q -p onnx-runtime-ep-cuda --features cuda,gpu-tests --depth 0
nvidia-smi -i "$ONNX_GENAI_CUDA_PHYSICAL_DEVICE" \
    --query-gpu=index,uuid,name,driver_version,pci.bus_id \
    --format=csv,noheader,nounits |
    sed 's/^/GPU_SCRIPT,logical_ordinal=0,physical=/'

LIST=$(cargo test -q --release -p onnx-runtime-ep-cuda --features cuda,gpu-tests \
    --test einsum_gpu einsum_captured_descriptor_benchmark \
    -- --ignored --exact --list 2>&1) || {
    echo "BUILD-FAILED: could not enumerate the benchmark test" >&2
    printf '%s\n' "$LIST" >&2
    exit 3
}
printf 'TEST_LIST,%s\n' "$(printf '%s\n' "$LIST" | tr '\n' '|')"
COUNT=$(printf '%s\n' "$LIST" | grep -c '^einsum_captured_descriptor_benchmark: test$' || true)
if [[ "$COUNT" != 1 ]]; then
    echo "FILTER-DRIFT: expected exactly one named benchmark, selected $COUNT" >&2
    exit 2
fi

cargo test -q --release -p onnx-runtime-ep-cuda --features cuda,gpu-tests \
    --test einsum_gpu einsum_captured_descriptor_benchmark \
    -- --ignored --exact --nocapture
