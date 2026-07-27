# Pris — Regression guard hardening

**Date:** 2026-07-27
**Author:** Pris
**Scope:** `crates/onnx-runtime-ep-cpu/src/kernels/matmul.rs`, `crates/onnx-genai-bench/tests/profile_native.rs`

## Problem

The throughput regression floor of 3.50 tok/s was the pre-campaign baseline and
could not catch a 4.5× regression from 60→13 tok/s. The same defect family
(threshold too low to fail) that Chew caught in GEMV tolerance and the reviewer
caught in the cache assertion.

## Dispatch reachability test

Added `fp16_m1_decode_reaches_neon_gemv_not_half_gemm` with an atomic
`GEMV_F16_TEST_HITS` counter (same pattern as `SDPA_NEON_TEST_HITS`). The test
creates f16×f16 M=1 tensors (matching real model dtype) and asserts the GEMV
path was reached, not `try_matmul_half`.

- Guard-break verified: on current HEAD (before Iran's M=1 gate), the test fails
  with the exact assertion: "half_gemm.rs is likely intercepting M=1".
- Counter and test properly cfg-gated: `#[cfg(all(target_arch = "aarch64",
  target_os = "macos"))]` — no dead code on x86_64.

## Throughput floors

**FP32** (measurement rig absolute / all-machine roofline):
- 3.50 → **18.0 tok/s** (54% of published 33.6; pre-campaign was 3.83)
- 0.30 → **0.35** roofline fraction

**FP16** (new — separate test):
- **28.0 tok/s** absolute (47% of quiet-host 60.41; the 4.5× regression was 13.37)
- **0.25** roofline fraction

Design: absolute floor on measurement rig catches catastrophic regressions; roofline
fraction on all machines normalizes away host-load and machine-bandwidth variance.
Both checked on the measurement rig for defense in depth.

## Why these numbers

Calibrated from 5-run medians on M1 Max under varying load:
- FP16: measured 34–49 tok/s (vs 60 quiet host); min 31. Floor 28 gives headroom.
- FP32: measured 20–30 tok/s (vs 33.6 published); min 19. Floor 18 gives headroom.
- Roofline fractions set 30–40% below worst-case measured to avoid flakiness.

The dispatch test is the sharp guard; the floor is the blunt safety net.
