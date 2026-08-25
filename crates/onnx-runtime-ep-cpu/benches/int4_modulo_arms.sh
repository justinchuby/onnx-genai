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

echo "--- built arms (distinct binaries are the first check that the patch took)"
sha256sum "$OUT"/prefill_* "$OUT"/decode_*
echo "--- now: int4_modulo_matrix.py --rounds 61"
