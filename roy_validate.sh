#!/usr/bin/env bash
# Roy's local reproduction of the CI gate matrix (ci.yml + miri.yml) plus the
# explicit cross-arch steps that scripts/check_cross_compile.sh does NOT cover
# on Linux (its TARGET is x86_64-unknown-linux-gnu == native there).
set -uo pipefail
cd "$(dirname "$0")"

PKGS_BUILD=$(sed -n '164,195p' .github/workflows/ci.yml | grep -o '\-p [a-z0-9-]*' | tr '\n' ' ')
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

pass=0; fail=0
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

step "A fmt-all"            cargo fmt --all -- --check
step "B build-offline"      bash -c "cargo build --locked $PKGS_BUILD"
step "C test-ep-cpu"        cargo test --locked -p onnx-runtime-ep-cpu
step "D clippy-offline"     bash -c "cargo clippy --locked --all-targets $PKGS_BUILD -- -D warnings"
step "E test-mlas-feature"  cargo test --locked -p onnx-runtime-ep-cpu --features mlas
step "F clippy-native-be"   cargo clippy --locked --all-targets -p onnx-genai-engine --features native-backend -- -D warnings
step "G clippy-aarch64-linux" cargo clippy --locked --target aarch64-unknown-linux-gnu -p onnx-runtime-ep-cpu -- -D warnings
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
echo "PASS=$pass FAIL=$fail"
