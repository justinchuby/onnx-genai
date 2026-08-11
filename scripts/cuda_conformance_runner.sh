#!/usr/bin/env bash
# CUDA EP Hardware Conformance Runner
#
# Run this on a host with an NVIDIA GPU to validate the CUDA execution provider.
# It detects preconditions and runs the full validation chain:
#   device enumeration → allocator → stream → data transfer → capability → compile → execute
#
# Exit codes:
#   0 = VALIDATED (all tests pass on real GPU hardware)
#   1 = FAILED (test failures on GPU hardware — real bugs)
#   2 = UNVALIDATED (preconditions not met — no GPU, no driver, or cuda feature not enabled)
#
# Usage:
#   ./scripts/cuda_conformance_runner.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/cuda_conformance_runner.sh
#
# Author: Pris (tester)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════"
echo "  CUDA EP Hardware Conformance Runner"
echo "  $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── Precondition checks ────────────────────────────────────────────────────

SKIP_REASON=""

# Check 1: nvidia-smi present and GPU detected
if ! command -v nvidia-smi &>/dev/null; then
    SKIP_REASON="nvidia-smi not found — no NVIDIA driver installed"
elif ! nvidia-smi &>/dev/null; then
    SKIP_REASON="nvidia-smi failed — GPU not available or driver not loaded"
fi

# Check 2: CUDA libraries loadable
if [ -z "$SKIP_REASON" ]; then
    if ! ldconfig -p 2>/dev/null | grep -q libcuda.so; then
        if [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ] && \
           [ ! -f /usr/local/cuda/lib64/libcuda.so ]; then
            SKIP_REASON="libcuda.so not found — CUDA driver library missing"
        fi
    fi
fi

# Check 3: Rust toolchain can build with cuda feature
if [ -z "$SKIP_REASON" ]; then
    if ! cargo check -p onnx-runtime-ep-cuda --features cuda --message-format=short 2>/dev/null; then
        SKIP_REASON="cargo check -p onnx-runtime-ep-cuda --features cuda failed — cuda feature not buildable on this host"
    fi
fi

# ─── Report skip if preconditions not met ────────────────────────────────────

if [ -n "$SKIP_REASON" ]; then
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  STATUS: UNVALIDATED                                        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Reason: $SKIP_REASON"
    echo ""
    echo "  The CUDA execution provider has NOT been validated on this host."
    echo "  This is NOT a test failure — it means the preconditions for GPU"
    echo "  testing are not met. Do not claim CUDA works without running"
    echo "  this script on a host with:"
    echo "    - NVIDIA GPU with compute capability ≥ 7.0"
    echo "    - NVIDIA driver ≥ 535.x"
    echo "    - libcuda.so / libcublas.so loadable"
    echo "    - Rust workspace building with 'cuda' feature"
    echo ""
    exit 2
fi

# ─── Preconditions met: run the validation suite ─────────────────────────────

echo -e "${GREEN}Preconditions met:${NC}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
echo ""

echo "Running CUDA EP conformance suite..."
echo ""

# Phase 1: Device/allocator/transfer tests
echo "── Phase 1: Device allocator + data transfer ──"
if ! cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test device_allocator_gpu -- --nocapture 2>&1; then
    echo -e "${RED}FAILED: Device allocator tests${NC}"
    exit 1
fi
echo ""

# Phase 2: Construction/movement (exercises stream + H2D/D2H/D2D)
echo "── Phase 2: Construction + movement (stream + copies) ──"
if ! cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test construction_gpu -- --nocapture 2>&1; then
    echo -e "${RED}FAILED: Construction/movement tests${NC}"
    exit 1
fi
echo ""

# Phase 3: Capability + compile + execute (matmul as canonical op)
echo "── Phase 3: MatMul capability → compile → execute ──"
if ! cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test matmul_gpu -- --nocapture 2>&1; then
    echo -e "${RED}FAILED: MatMul parity tests${NC}"
    exit 1
fi
echo ""

# Phase 4: Full conformance sweep (all claimed ops vs CPU oracle)
echo "── Phase 4: Full conformance sweep vs CPU oracle ──"
if ! cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test cuda_conformance_gpu -- --nocapture 2>&1; then
    echo -e "${RED}FAILED: Conformance sweep${NC}"
    exit 1
fi
echo ""

# Phase 5: Attention (critical path)
echo "── Phase 5: Attention kernels ──"
if ! cargo test -p onnx-runtime-ep-cuda --features cuda \
    --test attention_gpu -- --nocapture 2>&1; then
    echo -e "${RED}FAILED: Attention tests${NC}"
    exit 1
fi
echo ""

# ─── All passed ──────────────────────────────────────────────────────────────

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  STATUS: VALIDATED                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  All CUDA EP conformance phases passed on real GPU hardware."
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  Timestamp: $(date -Iseconds)"
echo ""
exit 0
