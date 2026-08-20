# Decision: the "fc1 loader blocker" / all-fp16 QMoE rejection is RESOLVED (native CUDA now runs all-fp16 fused QMoE)

**Author:** Copilot (coding agent)
**Date:** 2026-08-20
**Scope:** native CUDA `com.microsoft::QMoE`, fp16/bf16 fixtures
**Related:** corrects the phantom in `.squad/identity/now.md:71` and the stale
mixed-precision root-cause note in `.squad/decisions-archive/2026-08.md` (the
2026-08-03 "loader blocker" entry). Those are append-only history and are NOT
rewritten — this decision supersedes them.

## What was believed (now falsified)

Two stale framings existed:

1. A *loader* blocker: the fused fp16 QMoE artifact was thought to be a
   **mixed-precision** graph (fp16 activations + **fp32** scales/router_probs)
   that ORT rejects at `Session::new` because `com.microsoft::QMoE` binds
   `input`, `router_probs`, `scales`, biases and output to a single type
   parameter `T`.
2. That the *only* native-kernel blocker for an all-fp16 graph was
   `router_probs` hardcoded to fp32.

## What is actually true (measured, 2026-08-20)

- Verified against ORT's schema (`ContribOperators.md` / `quantization_defs`):
  `com.microsoft::QMoE` has a **single type constraint `T` = {float, float16,
  bfloat16}** covering input, router_probs, scales, biases and output. So a
  valid fp16 export carries fp16 router_probs **and** fp16 scales **and** fp16
  biases/aggregation weights.
- The current `qwen15-moe-qmoe-mobius` fixture is **internally type-consistent
  (all fp16)** — it already satisfies ORT's single-`T` rule. The Mobius export
  was fixed; the *loader* blocker is gone. `qwen15-moe-qmoe-f32` is all-f32.
- The remaining blocker was entirely in **our** native CUDA kernel
  (`crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs`), and it was **broader than
  router_probs**: `router_probs`, `fc1_scales`, `fc2_scales` (and, latently,
  fc3 scales, biases and the optional aggregation weights) were all hardcoded to
  require `Float32`. router_probs was merely the first check to fail; scales
  would have failed next.

## Fix

Per-operand widen of the `T`-typed float operands to f32 device scratch (an
exact fp16/bf16→f32 upcast), then reuse the **unchanged** f32 routing/dequant
kernels (which already accept fp16 *activations*). Design notes:

- **Accumulation precision unchanged:** routing (max / `expf` / sum / normalize
  / top-k) still runs in **f32** in `qmoe_route`; dequant scales are f32. No
  softmax/top-k is done in fp16.
- **f32 path is byte-identical:** f32 operands keep their original pointers and
  never widen. Measured: `qwen15-moe-qmoe-f32` greedy token ids are identical
  between the pre-change control binary (69329c900) and the fixed binary.
- **Backward compatible:** the classification is per-operand, so the existing
  synthetic "fp16 activations + f32 scales" graphs keep working unchanged; an
  all-fp16 graph now runs too. This is a strict superset of ORT's single-`T`
  rule (the native backend is the sole validator — it skips ORT `Session::new`).

## Evidence

- `qwen15-moe-qmoe-mobius` (all fp16) now loads + decodes coherent text natively
  on CUDA: `": I am trying to find the best way to get the current date in a
  format that is easy to read. I"`; CUDA graph capture succeeds
  (`captures=1 replays=22 fallbacks=0 invalidations=0`).
- Route-selection dump confirms non-degenerate routing (top-k experts vary
  widely across tokens/layers, spanning the full expert range — not collapsed).
- `qmoe_gpu` GPU suite: 33 passed / 0 failed / 0 ignored under
  `--features cuda,gpu-tests`, incl. a new all-fp16/bf16 regression
  (`qmoe_all_half_router_scales_bias_match_rounded_cpu`).

**Do not re-investigate an fp16 QMoE "loader blocker" — it no longer exists.**
