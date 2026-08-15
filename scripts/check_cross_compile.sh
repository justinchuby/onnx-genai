#!/usr/bin/env bash
# check_cross_compile.sh — Cross-target compilation check
#
# Verifies the offline crate set compiles cleanly for a target that changes
# BOTH target_arch AND target_os from macOS arm64, catching the full class of
# cfg-gating errors that the popular local recipe misses:
#
#   cargo clippy --target x86_64-apple-darwin --all-targets -- -D warnings
#
# That command changes target_arch (aarch64→x86_64) but leaves target_os as
# "macos".  Every cfg(target_os = "macos") block stays compiled, so os-gating
# errors are invisible.  PR #317 proved the gap: `is_undilated` was used only
# inside a cfg(target_os = "macos") block, producing an unused-variable error
# on Linux/Windows — but the x86_64-apple-darwin check reported clean.
#
# This is the FOURTH non-macOS compilation break to reach CI.  All were
# catchable with a genuine Linux target that exercises both dimensions:
#
#   Target: x86_64-unknown-linux-gnu
#   Effect: target_arch = "x86_64", target_os = "linux"
#
# On CI (ubuntu-latest) this is the native target — no cross-compilation
# overhead.  On macOS (local dev) this is a genuine cross-compile; see the
# host-detection logic below for scope limitations.
#
# The architecture dimension
# --------------------------
# The Linux target above pins target_arch = "x86_64", which is the SAME arch
# the x86_64 CI lanes already build.  So it cannot catch the mirror-image
# mistake: an item that is only referenced from inside a
# cfg(target_arch = "x86_64") block becomes dead code on aarch64, and CI
# builds with -D warnings.  #1037 proved this gap — it vectorised the
# activation family on AVX2+FMA and left `SIMD_MIN_LEN`,
# `vector_path_available` and 26 MLAS polynomial constants referenced only
# from x86_64-gated code, which broke BOTH ARM64 lanes at the compile step
# while every x86_64 job stayed green.
#
# So this script now runs a second pass:
#
#   Target: aarch64-unknown-linux-gnu
#   Effect: target_arch = "aarch64", target_os = "linux"
#
# Between the two passes, every crate is compiled with target_arch and
# target_os each differing from the x86_64-Linux lanes at least once.  The
# ARM64 lanes stay the execution signal; this is only the compile gate, and it
# costs seconds on the quality lane instead of minutes on a scarce ARM runner.
#
# Known gaps
# ----------
# 1. Cannot catch RUNTIME dispatch errors (e.g. a backend enum arm that
#    compiles but is never reached).  That is dispatch-reachability territory
#    (check_dispatch_reachability.py).
# 2. On macOS local dev, crates with FFI build scripts (ort-sys, cpuinfo) fail
#    cross-compilation because Linux system headers are unavailable.  The
#    script falls back to the ort-sys-free subset.  CI (Linux) runs the full
#    set.
# 3. Windows-specific cfg issues (target_os = "windows") are not exercised.
#    The portable test matrix covers Windows; this script covers "not macOS."
# 4. onnx-runtime-cpuinfo is excluded from the aarch64 pass: its cmake build
#    script compiles C for the target and so needs a cross gcc that neither
#    ubuntu-latest nor a macOS host has.  It contains no arch-gated Rust.
#
# Exit codes
# ----------
#   0 — all checked crates compiled cleanly
#   1 — compilation errors detected (the interesting case)
#   2 — setup failure (target not installable, cargo missing, etc.)

set -euo pipefail

TARGET="x86_64-unknown-linux-gnu"
ARCH_TARGET="aarch64-unknown-linux-gnu"

# The full offline crate set from CI — every crate whose normal+dev dependency
# tree contains no ort-sys/CUDA dependency that needs a native toolchain.
# Matches ci.yml lines 91–118 minus mlas-sys (Linux-only, needs native gcc).
CRATES_FULL=(
    onnx-genai-comfyui-config
    onnx-genai-metadata
    onnx-genai-genai-config
    onnx-genai-kv
    onnx-genai-runtime-config
    onnx-genai-scheduler
    onnx-runtime-protocol-trace
    onnx-runtime-ep-api
    onnx-runtime-ep-cpu
    onnx-runtime-ir
    onnx-runtime-optimizer
    onnx-runtime-loader
    onnx-runtime-shape-inference
    onnx-runtime-quantization
    onnx-runtime-tracer
    onnx-runtime-session
    onnx-genai-preprocess
    onnx-std
    onnx-genai-router
    onnx-runtime-memory
    onnx-runtime-cpuinfo
    onnx-runtime-eager
    onnx-runtime-capi
    onnx-runtime-dlpack
    onnx-runtime-comm
    onnx-std-python
)

# Subset that compiles without ort-sys or cmake-based build scripts.
# These crates have no FFI build dependencies that require cross-linker
# or system headers for the target.  On macOS this is the best we can do
# without installing a Linux sysroot.
#
# Excluded from the full set:
#   onnx-runtime-ep-api, onnx-runtime-ep-cpu — depend on onnx-genai-ort-sys
#   onnx-runtime-session, onnx-runtime-capi  — depend on onnx-genai-ort-sys
#   onnx-runtime-eager                       — depends on onnx-genai-ort-sys
#   onnx-runtime-cpuinfo                     — cmake build script needs gcc
#   mlas-sys                                 — Linux-only native build
CRATES_NO_FFI=(
    onnx-genai-comfyui-config
    onnx-genai-metadata
    onnx-genai-genai-config
    onnx-genai-kv
    onnx-genai-runtime-config
    onnx-genai-scheduler
    onnx-runtime-protocol-trace
    onnx-runtime-ir
    onnx-runtime-optimizer
    onnx-runtime-loader
    onnx-runtime-shape-inference
    onnx-runtime-quantization
    onnx-runtime-tracer
    onnx-genai-preprocess
    onnx-std
    onnx-genai-router
    onnx-runtime-memory
    onnx-runtime-dlpack
    onnx-runtime-comm
    onnx-std-python
)

# ─── Target installation ───────────────────────────────────────────────────

for t in "$TARGET" "$ARCH_TARGET"; do
    if ! rustup target list --installed | grep -q "^$t\$"; then
        echo "▶ Installing cross-target $t (one-time setup)..."
        if ! rustup target add "$t" 2>/dev/null; then
            echo "✗ Failed to install target $t." >&2
            echo "  Run: rustup target add $t" >&2
            exit 2
        fi
    fi
done

# ─── Host detection and scope selection ────────────────────────────────────

HOST_OS="$(uname -s)"
if [ "$HOST_OS" = "Linux" ]; then
    # On Linux, x86_64-unknown-linux-gnu is the native target (or at worst a
    # same-OS cross from aarch64).  All crates compile without a foreign
    # sysroot.  Use the full set.
    CRATES=("${CRATES_FULL[@]}")
    SCOPE_NOTE="full offline set (native target)"
    # Same-OS cross, so no foreign sysroot is needed and every crate that is
    # pure Rust (or whose build script only generates Rust) still checks.  Only
    # cpuinfo, which compiles C for the target, needs a cross gcc.
    ARCH_CRATES=()
    for crate in "${CRATES_FULL[@]}"; do
        [ "$crate" = "onnx-runtime-cpuinfo" ] && continue
        ARCH_CRATES+=("$crate")
    done
    ARCH_SCOPE_NOTE="full offline set minus onnx-runtime-cpuinfo (needs a cross gcc)"
else
    # On macOS (or other non-Linux hosts), ort-sys and cpuinfo need Linux
    # system headers that are unavailable without a cross-sysroot.  Fall back
    # to the FFI-free subset.  This still catches os-gating errors in the IR,
    # optimizer, and loader crates — and CI (Linux) covers the rest.
    CRATES=("${CRATES_NO_FFI[@]}")
    SCOPE_NOTE="FFI-free subset (ort-sys/cpuinfo excluded — CI covers the full set)"
    ARCH_CRATES=("${CRATES_NO_FFI[@]}")
    ARCH_SCOPE_NOTE="$SCOPE_NOTE"
    echo "⚠  Running on $HOST_OS — scoping to crates without FFI build scripts."
    echo "   To check onnx-runtime-ep-cpu locally, install a Linux sysroot or"
    echo "   rely on CI (ubuntu-latest) where this script runs the full set."
    echo ""
fi

# ─── Build the -p flags ───────────────────────────────────────────────────

PKGS=()
for crate in "${CRATES[@]}"; do
    PKGS+=("-p" "$crate")
done

ARCH_PKGS=()
for crate in "${ARCH_CRATES[@]}"; do
    ARCH_PKGS+=("-p" "$crate")
done

# ─── Run the check ────────────────────────────────────────────────────────

echo "▶ Cross-compile check (target: $TARGET, scope: $SCOPE_NOTE)"
echo "  Command: cargo clippy --target $TARGET --all-targets ${PKGS[*]} -- -D warnings"
echo ""

if ! cargo clippy --locked --target "$TARGET" --all-targets "${PKGS[@]}" -- -D warnings; then
    echo "" >&2
    echo "✗ Cross-compile check FAILED." >&2
    echo "" >&2
    echo "  One or more crates do not compile for $TARGET." >&2
    echo "" >&2
    echo "  WHY THIS CHECK EXISTS:" >&2
    echo "  The common local recipe:" >&2
    echo "    cargo clippy --target x86_64-apple-darwin --all-targets -- -D warnings" >&2
    echo "  changes target_arch but leaves target_os = \"macos\".  It CANNOT catch" >&2
    echo "  cfg(target_os = \"macos\") errors.  PR #317 proved this: \`is_undilated\`" >&2
    echo "  was used only inside a macOS cfg block, so x86_64-apple-darwin reported" >&2
    echo "  clean while Linux/Windows CI failed with 'unused variable'." >&2
    echo "" >&2
    echo "  FIX: ensure variables/imports used inside cfg(target_os = \"macos\") blocks" >&2
    echo "  are themselves gated, or used unconditionally." >&2
    exit 1
fi

echo ""
echo "✓ OS-dimension check passed ($SCOPE_NOTE)"

# ─── Run the architecture check ───────────────────────────────────────────

echo ""
echo "▶ Cross-arch check (target: $ARCH_TARGET, scope: $ARCH_SCOPE_NOTE)"
echo "  Command: cargo clippy --target $ARCH_TARGET --all-targets ${ARCH_PKGS[*]} -- -D warnings"
echo ""

if ! cargo clippy --locked --target "$ARCH_TARGET" --all-targets "${ARCH_PKGS[@]}" -- -D warnings; then
    echo "" >&2
    echo "✗ Cross-arch check FAILED." >&2
    echo "" >&2
    echo "  One or more crates do not compile for $ARCH_TARGET." >&2
    echo "" >&2
    echo "  WHY THIS CHECK EXISTS:" >&2
    echo "  Every other blocking lane builds target_arch = \"x86_64\", so an item" >&2
    echo "  referenced ONLY from inside a cfg(target_arch = \"x86_64\") block looks" >&2
    echo "  used everywhere it is checked — and is dead code on ARM64, which CI" >&2
    echo "  builds with -D warnings.  #1037 proved this: simd_activations.rs left" >&2
    echo "  SIMD_MIN_LEN, vector_path_available and 26 MLAS polynomial constants" >&2
    echo "  reachable only from x86_64 code, breaking both ARM64 lanes at the" >&2
    echo "  compile step while every x86_64 job stayed green." >&2
    echo "" >&2
    echo "  FIX: gate x86-only items with cfg(target_arch = ...) alongside their" >&2
    echo "  only consumer, or — when portable tests still need them — mark them" >&2
    echo "  #[cfg_attr(not(target_arch = \"x86_64\"), allow(dead_code))]." >&2
    exit 1
fi

echo ""
echo "✓ Cross-arch check passed ($ARCH_SCOPE_NOTE)"
echo ""
echo "✓ Cross-compile check passed (target_os and target_arch dimensions)"
