# Decision: Feature Gate Coverage Lint

**Author:** Resch  
**Date:** 2026-07-28  
**Status:** Proposed (PR pending)

## Context

Instance #14 of the "optimization exists and never executes" campaign: Clip was
76.8% of MobileNetV2 runtime because its fast path existed only behind
`cfg(feature = "mlas")`. Without MLAS (which is unreachable on macOS/aarch64 by
construction), every Clip allocated a Vec, copied element-by-element, clamped,
and copied back — two full copies per node, 35 nodes, ~42 MB of wasted
bandwidth per inference.

This is a repeat of instance #8 (conv_ref.rs scalar-only on macOS). The five
existing CI layers did not catch it because they are all anchored on **counters
that exist**. An op with a slow path and no counter at all is invisible.

## Decision

Add `scripts/check_feature_gate_coverage.py` — a sixth CI layer that
specifically targets the blind spot: a cfg-gated performance path in a kernel
`execute` function whose fallback is unmonitored (no platform-specific fast path
AND no _TEST_HITS counter).

## Audit findings

### MLAS-gated ops — full inventory

| Op | File | Fallback without MLAS | Severity | Models affected |
|---|---|---|---|---|
| **Clip** | selection.rs:31 | `to_dense` → scalar clamp → `write_dense` (double copy) | **CRITICAL** | MobileNetV2 (76.8% of runtime, 35 nodes) |
| **Relu** | relu.rs:38 | `to_dense_f32_widen` → `relu_in_place` → `write_dense_f32_narrow` (double copy) | **HIGH** | ResNet (many Relu nodes after Conv) |
| **GlobalPool** | pooling.rs:481 | `to_dense_f32_widen` → scalar iteration → `write_dense_f32_narrow` | **MEDIUM** | MobileNetV2, ResNet (1 node each, small spatial at that point) |
| Pool (regular) | pooling.rs:311 | Has BNNS fast path on macOS ✓ | OK | — |
| Add | add.rs:96 | Has vDSP fast path on macOS ✓ | OK | — |
| SiLU | activations.rs:325 | Has NEON vectorized path ✓ | OK | — |
| MatMul (GEMM) | matmul.rs:598 | Has Accelerate/NEON/SimdX86 backends ✓ | OK | — |
| MatMul (packed_b) | matmul.rs:1536 | Falls through to gemm_with_backend ✓ | OK | — |
| MatMulNBits (SQNBit) | matmul_nbits.rs:538 | Has packed int4/int8 GEMV paths ✓ | OK | — |
| SDPA | sdpa.rs:296 | Has Accelerate + NEON paths ✓ | OK | — |

### Beyond MLAS — other unreachable gates

The `target_arch = "x86_64"` gates in `block_quantized_matmul.rs` use **runtime**
feature detection (`is_x86_feature_detected!`) — always compiled, correct pattern.

The `target_os = "macos"` gates (Accelerate, BNNS, vDSP) are reachable on macOS
and have scalar fallbacks for Linux. These are the *inverse* of the MLAS problem:
Linux lacks the macOS frameworks, but on Linux MLAS provides the fast path.
Cross-coverage is complete.

No other feature flags guard performance paths in the kernel directory.

### Impact ranking (Amdahl discipline)

1. **Clip** — CRITICAL. 35 nodes in MobileNetV2, large spatial tensors. 76.8% of model runtime. **3.9× model speedup from fix.**
2. **Relu** — HIGH. Numerous in ResNet/classifier models (after every Conv). Same double-copy mechanism as Clip but Relu nodes operate on post-Conv tensors (typically smaller spatial than Clip's input range). Estimated 15-30% of model runtime in ResNet without BNNS Conv.
3. **GlobalPool** — MEDIUM. 1 node per model, operates on small spatial tensors at the end of the network. Scalar iteration on ~7×7 spatial is negligible. Not worth a dedicated SIMD path.

## What the check cannot catch

Stated in the script's docstring (known gaps 1–5). The structural limitation:
this lint catches missing *instrumentation* on fallback paths, not missing
*optimizations*. A kernel author who adds a TEST_HITS counter to a pathologically
slow fallback satisfies the lint — but the manifest then makes the tier visible,
and a human can ask "why is this tier3 on macOS?". The layered defense works by
making silence impossible, not by automating performance engineering.
