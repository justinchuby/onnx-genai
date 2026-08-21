#!/usr/bin/env bash
# Roy's local reproduction of the CI gate matrix (ci.yml + miri.yml) plus the
# explicit cross-arch steps that scripts/check_cross_compile.sh does NOT cover
# on Linux (its TARGET is x86_64-unknown-linux-gnu == native there).
set -uo pipefail
cd "$(dirname "$0")"

PKGS_BUILD=$(sed -n '164,195p' .github/workflows/ci.yml | grep -o '\-p [a-z0-9-]*' | tr '\n' ' ')
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

pass=0; fail=0; skip=0
step() {
  local name="$1"; shift
  echo "════════ $name ════════"
  if "$@" > ".rv_$(echo "$name" | tr ' /' '__').log" 2>&1; then
    echo "  PASS: $name"; pass=$((pass+1))
  else
    echo "  FAIL: $name  (see .rv_$(echo "$name" | tr ' /' '__').log)"; fail=$((fail+1))
    tail -25 ".rv_$(echo "$name" | tr ' /' '__').log"
  fi
}

# A step that needs a toolchain which may be absent. It is never silently
# dropped: an absent toolchain is reported as SKIP with the reason and the
# install hint, and counted separately, so "PASS=n FAIL=0" can never be read as
# "the matrix ran" when part of it did not.
step_if() {
  local name="$1" probe="$2" hint="$3"; shift 3
  if command -v "$probe" > /dev/null 2>&1; then
    step "$name" "$@"
  else
    echo "════════ $name ════════"
    echo "  SKIP: $name -- '$probe' not found. Install: $hint"
    skip=$((skip+1))
  fi
}

step "A fmt-all"            cargo fmt --all -- --check
step "B build-offline"      bash -c "cargo build --locked $PKGS_BUILD"
step "C test-ep-cpu"        cargo test --locked -p onnx-runtime-ep-cpu
step "D clippy-offline"     bash -c "cargo clippy --locked --all-targets $PKGS_BUILD -- -D warnings"
step "E test-mlas-feature"  cargo test --locked -p onnx-runtime-ep-cpu --features mlas
step "F clippy-native-be"   cargo clippy --locked --all-targets -p onnx-genai-engine --features native-backend -- -D warnings
step "G clippy-aarch64-linux" cargo clippy --locked --target aarch64-unknown-linux-gnu -p onnx-runtime-ep-cpu -- -D warnings
# G only *compiles* for aarch64. This runs the suite, so the non-x86 fallback
# paths are actually executed rather than merely type-checked. QEMU_LD_PREFIX
# (not `qemu -L`) is required because several tests re-exec `current_exe()` and
# the binfmt-launched child does not inherit a `-L` sysroot -- without it five
# affinity/SPMD tests fail with "Could not open '/lib/ld-linux-aarch64.so.1'",
# which looks like a code defect and is not one.
step_if "G2 test-aarch64-qemu" qemu-aarch64-static "apt install qemu-user-static" \
  env QEMU_LD_PREFIX=/usr/aarch64-linux-gnu \
      CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER=qemu-aarch64-static \
      cargo test --locked --target aarch64-unknown-linux-gnu -p onnx-runtime-ep-cpu --lib
step "H check-win-arm64"    cargo check --locked --target aarch64-pc-windows-msvc -p onnx-runtime-ep-cpu
step "I ep-cpu-no-default"  cargo test --locked -p onnx-runtime-ep-cpu --no-default-features
step "J ep-cpu-all-features" cargo check --locked -p onnx-runtime-ep-cpu --all-features
step "K no-mlas-artifacts"  cargo test --locked -p onnx-runtime-ep-cpu-plugin --test default_artifacts_are_mlas_free
step "L cross-compile-sh"   bash scripts/check_cross_compile.sh

for s in check_publish_order check_profile_table check_platform_naming \
         check_dispatch_reachability check_dispatch_manifest \
         check_feature_gate_coverage; do
  step "S $s" python3 "scripts/$s.py"
done
step "S verify_documented_env_vars" python3 .github/scripts/verify_documented_env_vars.py
step "S workspace_test_packages"    python3 .github/scripts/workspace_test_packages.py verify

echo "════════════════════════════════════"
echo "PASS=$pass FAIL=$fail SKIP=$skip"
if [ "$skip" -gt 0 ]; then
  echo "NOTE: $skip step(s) were skipped for a missing toolchain -- this matrix"
  echo "      is NOT complete. See the SKIP lines above for what to install."
fi
# Known environmental failure on a Linux host: step H cross-compiles to
# aarch64-pc-windows-msvc, which needs the MSVC/Windows SDK headers that only
# exist on a Windows runner (CI runs that gate natively on Windows). It fails
# inside onnx-genai-ort-sys' bindgen with "'stdlib.h' file not found", before
# reaching any workspace kernel code, and it fails identically on an unmodified
# main. Expect PASS=19 FAIL=1 here, not PASS=20.
