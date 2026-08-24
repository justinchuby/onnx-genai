#!/usr/bin/env bash
# check_cross_compile_test.sh — conformance suite for the submodule precondition
#
# `check_cross_compile.sh` is a gate, so the interesting question about its new
# precondition is not "does it print something" but "does it refuse BEFORE it
# spends anything, and does it stay silent on a correct checkout".  Neither can
# be answered by reading the script, and neither should be answered by breaking
# the real submodule on a developer's box.
#
# So every cell here runs the real script against a fixture repository root
# with `cargo` and `rustup` replaced by shims.  The shims make the whole run
# take about a second, and — more importantly — they RECORD being called, which
# is what lets the ordering cell assert that a broken checkout is rejected with
# nothing built rather than after a clippy pass.
#
# One cell deliberately runs against this repository's real root: fixtures can
# only prove the check reacts to a directory, not that it is looking at the
# directory cargo will actually build.  Without it every assertion below could
# pass while the gate rejected every correct checkout.
#
# Linux/macOS shell; no /proc, no GNU-only tools.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REAL_ROOT="$(pwd)"
SCRIPT="$REAL_ROOT/scripts/check_cross_compile.sh"
WORK="$REAL_ROOT/.crosscompile-selftest"
VENDOR_REL="crates/onnx-runtime-cpuinfo/vendor/cpuinfo"

PASS=0
FAIL=0

cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT

ok() {
    PASS=$((PASS + 1))
    printf '  ok   %s\n' "$1"
}

bad() {
    FAIL=$((FAIL + 1))
    printf '  FAIL %s\n' "$1"
    [ $# -gt 1 ] && printf '       %s\n' "$2"
}

check() {
    # check <description> <condition-as-exit-status-of-caller>
    if [ "$2" = "0" ]; then ok "$1"; else bad "$1" "${3:-}"; fi
}

# ─── Shims ────────────────────────────────────────────────────────────────
# `cargo` and `rustup` are replaced so no cell compiles anything.  Both append
# to $WORK/calls, which is how the ordering cell distinguishes "refused before
# doing any work" from "refused after a full clippy pass".

make_shims() {
    mkdir -p "$WORK/bin"
    cat > "$WORK/bin/rustup" <<'EOF'
#!/usr/bin/env bash
echo "rustup $*" >> "$SHIM_CALLS"
# Report every target as already installed so the script never tries to add one.
if [ "${1:-}" = "target" ] && [ "${2:-}" = "list" ]; then
    echo "x86_64-unknown-linux-gnu"
    echo "aarch64-unknown-linux-gnu"
fi
exit 0
EOF
    cat > "$WORK/bin/cargo" <<'EOF'
#!/usr/bin/env bash
echo "cargo $*" >> "$SHIM_CALLS"
exit 0
EOF
    chmod +x "$WORK/bin/rustup" "$WORK/bin/cargo"
}

# fixture_root <name> <file>... — a fake repo root holding a copy of the real
# script and a vendor tree containing exactly the named files.
fixture_root() {
    local name="$1"
    shift
    local root="$WORK/$name"
    mkdir -p "$root/scripts" "$root/$VENDOR_REL"
    cp "$SCRIPT" "$root/scripts/"
    local f
    for f in "$@"; do
        mkdir -p "$(dirname "$root/$VENDOR_REL/$f")"
        : > "$root/$VENDOR_REL/$f"
    done
    printf '%s' "$root"
}

# run_gate <root> — run the script inside <root> with the shims on PATH.
# Sets OUT (merged stdout+stderr), STATUS and CALLS.
run_gate() {
    local root="$1"
    : > "$WORK/calls"
    OUT="$(cd "$root" && PATH="$WORK/bin:$PATH" SHIM_CALLS="$WORK/calls" \
        bash "$root/scripts/check_cross_compile.sh" 2>&1)"
    STATUS=$?
    CALLS="$(cat "$WORK/calls")"
}

rm -rf "$WORK"
make_shims

echo "== an unpopulated vendor tree is refused, and says why =="

ROOT="$(fixture_root empty)"
run_gate "$ROOT"

check "exit status is 2 (setup fault), not 1 (compile error)" \
    "$([ "$STATUS" = "2" ] && echo 0 || echo 1)" "got $STATUS"

case "$OUT" in
    *"git submodule update --init crates/onnx-runtime-cpuinfo/vendor/cpuinfo"*) R=0 ;;
    *) R=1 ;;
esac
check "the exact fix command is printed" "$R" "$OUT"

case "$OUT" in *"git worktree add"*) R=0 ;; *) R=1 ;; esac
check "the cause is named, not only the cure" "$R"

case "$OUT" in *"$ROOT/$VENDOR_REL"*) R=0 ;; *) R=1 ;; esac
check "the directory it looked in is named" "$R" "$OUT"

case "$OUT" in *"environment fault"*) R=0 ;; *) R=1 ;; esac
check "the fault is classified as environmental" "$R"

# The whole argument for a precondition rather than relying on the build
# script's own message is that it costs nothing to reach.  If cargo has already
# run, this cell has no reason to exist.
case "$CALLS" in *cargo*) R=1 ;; *) R=0 ;; esac
check "refuses before invoking cargo at all" "$R" "calls: $CALLS"

echo ""
echo "== a half-populated tree is refused too, naming only what is missing =="

ROOT="$(fixture_root half CMakeLists.txt)"
run_gate "$ROOT"

check "exit status is 2" "$([ "$STATUS" = "2" ] && echo 0 || echo 1)" "got $STATUS"

case "$OUT" in *"include/cpuinfo.h"*) R=0 ;; *) R=1 ;; esac
check "the missing header is named" "$R" "$OUT"

case "$OUT" in *"missing:   CMakeLists.txt"*) R=1 ;; *) R=0 ;; esac
check "a file that is present is not listed as missing" "$R" "$OUT"

echo ""
echo "== a populated tree passes, silently =="

ROOT="$(fixture_root full CMakeLists.txt include/cpuinfo.h)"
run_gate "$ROOT"

check "exit status is 0" "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"

case "$OUT" in *"not populated"*) R=1 ;; *) R=0 ;; esac
check "says nothing about submodules" "$R" "$OUT"

# Negative control for the ordering cell above: if the shims were never
# reached in ANY cell, "refuses before invoking cargo" would pass vacuously.
case "$CALLS" in *"cargo clippy"*) R=0 ;; *) R=1 ;; esac
check "a passing run does reach cargo (so the ordering cell is not vacuous)" \
    "$R" "calls: $CALLS"

echo ""
echo "== the real repository root passes the precondition =="

# Anti-vacuity for the whole file.  Every cell above builds its own vendor
# tree, so all of them would still pass if the check looked at a path that no
# longer exists in this repository — and the gate would then reject every
# correct checkout.  This is the only cell that reads the tree cargo builds.
run_gate "$REAL_ROOT"

check "exit status is 0 on this checkout" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"

case "$OUT" in *"not populated"*) R=1 ;; *) R=0 ;; esac
check "this checkout is not reported as unpopulated" "$R" "$OUT"

echo ""
echo "== the precondition follows the script, not the caller's directory =="

# The gate resolves its own repository root from BASH_SOURCE, so running it
# from elsewhere must still check the tree it belongs to.  A relative path
# would silently report a fault for every caller outside the root.
ROOT="$(fixture_root fromelsewhere CMakeLists.txt include/cpuinfo.h)"
mkdir -p "$WORK/elsewhere"
: > "$WORK/calls"
OUT="$(cd "$WORK/elsewhere" && PATH="$WORK/bin:$PATH" SHIM_CALLS="$WORK/calls" \
    bash "$ROOT/scripts/check_cross_compile.sh" 2>&1)"
STATUS=$?

check "exit status is 0 when run from another directory" \
    "$([ "$STATUS" = "0" ] && echo 0 || echo 1)" "got $STATUS: $OUT"

echo ""
EXPECTED=15
TOTAL=$((PASS + FAIL))
if [ "$TOTAL" -ne "$EXPECTED" ]; then
    echo "✗ $TOTAL assertions ran, expected $EXPECTED — a cell was added or lost" >&2
    exit 1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "✗ $FAIL of $TOTAL assertions failed" >&2
    exit 1
fi

echo "✓ $PASS/$TOTAL assertions passed"
