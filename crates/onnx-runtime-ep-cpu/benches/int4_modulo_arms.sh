#!/usr/bin/env bash
# Build the three arms of the dequant_panel_avx2 modulo-elimination A/B from a
# single source tree, copying each binary out so the timed matrix can interleave
# arms without rebuilding between reps.
#
# The arms differ by exactly one line of `Int4Weight::dequant_panel_avx2`:
#
#   after   let offset_in_block = offset_base + q;              (main, #1809)
#   before  let offset_in_block = (depth + q) % block_size;     (pre-#1809)
#   poison  let offset_in_block = offset_base;                  (route proof)
#
# The poison drops the `+ q` term. It is always in bounds -- `offset_base` is
# the value the q = 0 iteration legitimately uses -- but wrong for every q > 0,
# so any row whose route reaches this line moves its checksum, and any row that
# does not is left bit-identical as a built-in control.
#
# INDEPENDENT LAYOUT. One pair of binaries has one code layout, and no
# same-binary A/A can see the difference between two of them. To get a second,
# independent layout on demand, rebuild every arm with function alignment
# forced. It changes nothing an instruction executes; it moves where functions
# start (~50% -> ~91% of FUNC symbols on a 32-byte boundary here):
#
#   RUSTFLAGS=-Cllvm-args=-align-all-functions=5 \
#   MOD_ARMS_OUT=target/int4-modulo-arms-align32 \
#       crates/onnx-runtime-ep-cpu/benches/int4_modulo_arms.sh
#
# Better than re-running the A/B under it: point the matrix at a directory whose
# `before` is the default-layout `after` and whose `after` is the aligned one.
# The headline ratio is then a pure A/B' layout null -- same source line, two
# layouts -- and the harness's own bit-identity check across arms proves the
# semantics really were held constant while only the layout moved.
#
# Measured that way through the SMT-sibling gate (#2216): 1.0014 [1.0000,
# 1.0042] at prefill block 32 m=1 and 0.9993 [0.9971, 1.0005] on the block 16
# decode. Both null. Layout sensitivity at these cells is under half a percent.
#
# This header used to say the layout component "reaches ~2%", read off the
# spread across three build pairs taken before that gate existed. The direct
# measurement says that spread was contention, not layout. Do not carry 2%
# forward as a credibility bar; measure the null for the cell you care about.
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SRC=crates/onnx-runtime-ep-cpu/src/kernels/x86_sgemm.rs
OUT=${MOD_ARMS_OUT:-target/int4-modulo-arms}
AFTER='                let offset_in_block = offset_base + q;'
BEFORE='                let offset_in_block = (depth + q) % block_size;'
POISON='                let offset_in_block = offset_base;'

# This script patches a tracked source file in place and restores it with
# `git checkout --`, which would silently discard uncommitted work. Refuse
# rather than do that.
if ! git diff --quiet -- "$SRC" || ! git diff --cached --quiet -- "$SRC"; then
    echo "refusing: $SRC has uncommitted changes, and this script restores it" \
         "with \`git checkout --\`, which would discard them." >&2
    exit 1
fi

mkdir -p "$OUT"
trap 'git checkout -- "$SRC"' EXIT

newest_exe() {
    find target/release/deps -maxdepth 1 -type f -executable -name "$1-*" \
        ! -name '*.d' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-
}

build_arm() {
    local name=$1 line=$2
    python3 - "$SRC" "$AFTER" "$line" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if s.count(old) != 1:
    raise SystemExit(f"expected exactly one {old!r}, found {s.count(old)}")
open(path, "w").write(s.replace(old, new))
PY
    # Touch is not enough on its own -- assert the source really says what this
    # arm claims before the build, so a failed patch cannot silently produce a
    # duplicate of the previous arm.
    grep -qF "$line" "$SRC" || { echo "arm $name did not patch" >&2; exit 1; }
    cargo bench -p onnx-runtime-ep-cpu \
        --bench int4_prefill_route_ab --bench int4_decode_loop_ab --no-run 2>&1 | tail -3
    # The glob also matches `.d` dep-files; take the newest executable.
    cp "$(newest_exe int4_prefill_route_ab)" "$OUT/prefill_$name"
    cp "$(newest_exe int4_decode_loop_ab)" "$OUT/decode_$name"
    git checkout -- "$SRC"
}

build_arm after "$AFTER"
build_arm before "$BEFORE"
build_arm poison "$POISON"

# The null arm. A separate file rather than a second run of `after`, so it is a
# genuinely independent launch and pays every per-launch cost the real arms pay
# -- ASLR, page backing, first touch. A null taken any other way understates the
# noise floor it exists to bound.
cp "$OUT/prefill_after" "$OUT/prefill_aa"
cp "$OUT/decode_after" "$OUT/decode_aa"

echo "--- built arms"
sha256sum "$OUT"/prefill_* "$OUT"/decode_*

# Hard gate, not a printout to eyeball. Two arms with the same bytes is the
# worst outcome this script has: the matrix still runs, every row still reports
# a number, and the number is a null that means nothing because both sides were
# the same binary. It is reachable without anyone doing anything wrong -- the
# first arm patches the source to the line that is already on `main`, so if
# cargo ever decides that write was a no-op (mtime granularity, a cached
# fingerprint) it will skip the rebuild and `newest_exe` will hand back the
# previous arm. `aa` is excluded because it is a deliberate copy of `after`.
for kind in prefill decode; do
    dupes=$(sha256sum "$OUT/${kind}_before" "$OUT/${kind}_after" "$OUT/${kind}_poison" \
        | awk '{print $1}' | sort | uniq -d)
    if [ -n "$dupes" ]; then
        echo "FAILED: two $kind arms are byte-identical, so at least one did not" \
             "rebuild. Any matrix taken from these is a null between a binary" \
             "and itself. Try \`cargo clean -p onnx-runtime-ep-cpu\`." >&2
        exit 1
    fi
done
echo "--- all three arms are distinct binaries"
echo "--- now: int4_modulo_matrix.py --rounds 61"
