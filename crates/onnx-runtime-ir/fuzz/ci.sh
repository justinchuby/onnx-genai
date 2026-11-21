#!/usr/bin/env bash
# Bounded fuzz lane for the IR pattern parser/compiler, run by
# .github/workflows/fuzz.yml and reproducible locally with the same command:
#
#   crates/onnx-runtime-ir/fuzz/ci.sh [seconds-per-target]
#
# Requires a nightly toolchain and cargo-fuzz. Two phases per target:
#   1. replay every committed seed (regressions included) exactly once;
#   2. coverage-guided smoke for SECONDS from those seeds.
# Committed seeds are read-only: new units go to a scratch corpus under
# fuzz/corpus/. A crash, timeout or RSS overrun writes a reproducer to
# fuzz/artifacts/<target>/ and fails the script.
set -euo pipefail

seconds="${1:-60}"
if ! [[ "$seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "seconds-per-target must be a positive integer, got '$seconds'" >&2
    exit 2
fi
cd "$(dirname "$0")"
toolchain="${FUZZ_TOOLCHAIN:-nightly}"
targets=(pattern_text pattern_tokens)
limits=(-rss_limit_mb=2048 -timeout=10)

cargo "+$toolchain" fuzz build -O -a
for target in "${targets[@]}"; do
    mkdir -p "corpus/$target" "artifacts/$target"
    echo "::group::$target: replay committed seeds"
    # File arguments avoid libFuzzer's corpus max_len truncation.
    cargo "+$toolchain" fuzz run -O -a "$target" "seeds/$target/"* -- -runs=1 "${limits[@]}"
    echo "::endgroup::"
    echo "::group::$target: ${seconds}s coverage-guided smoke"
    cargo "+$toolchain" fuzz run -O -a "$target" "corpus/$target" "seeds/$target" -- \
        -max_total_time="$seconds" -print_final_stats=1 -max_len=8192 "${limits[@]}"
    echo "::endgroup::"
done
