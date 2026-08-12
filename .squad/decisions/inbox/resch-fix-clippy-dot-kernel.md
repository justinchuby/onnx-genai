# Decision: Fix arch-gated `dot_kernel` unused-param clippy error blocking CI

- **Author:** Resch (Intel CPU Optimization Engineer)
- **Date:** 2026-08-12
- **Branch:** `squad/fix-clippy-dot-kernel` (based on origin/main @ 514181d6)
- **Scope:** `crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs`

## Problem

Main's CI quality lane (Linux x86_64, `RUSTFLAGS="-D warnings" cargo clippy
--all-targets -- -D warnings`) was RED, blocking every PR. Root cause: fn
`borrowed_affine_int4_matmul` (~line 4898) takes a `dot_kernel: DotKernel`
parameter that is referenced **only** inside `#[cfg(target_arch = "aarch64")]`
blocks (the `m == 1` NEON-dot fast path). On x86_64 the parameter is unused →
`unused_variables` → hard error under `-D warnings`.

## Fix (minimal, DRY, no behavior change)

Added `let _ = dot_kernel;` at the top of the function body, with a short
explanatory comment. This:

- Silences the x86_64 (and any non-aarch64) unused-variable error.
- Keeps `dot_kernel` genuinely used on aarch64 (`DotKernel` is `Copy`, so the
  discard is a no-op and the later `matches!(dot_kernel, DotKernel::NeonDot)`
  guards still work).
- **Mirrors the existing in-file convention**: three sibling matmul helpers
  already use `let _ = dot_kernel;` for the same arch-gating reason
  (matmul_nbits.rs lines ~3779, ~4272, ~4448). Chose this over
  `#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]` for
  consistency with established code — it's the DRY idiom this file already uses.

No numerical behavior change; no new scope. `borrowed_affine_int4_matmul` was
the only function missing the silencer (the other three arch-gated helpers
already had it) — file scan confirmed no other `dot_kernel`-style unused items.

## Verification (exactly as CI)

```
RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p onnx-runtime-ep-cpu -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 16.77s   # clean, exit 0

RUSTFLAGS="-D warnings" cargo clippy --locked --all-targets -p mlas-sys -p onnx-runtime-ep-api -- -D warnings
    Finished `dev` profile ... in 1m 12s                                   # clean, exit 0
```

`cargo fmt --all` applied (no additional changes). ort-sys-dependent crates
skipped (offline lane doesn't build them).

## Diff

```diff
@@ fn borrowed_affine_int4_matmul(
     debug_assert_eq!(activations.len(), m * k);
     debug_assert_eq!(result.len(), m * n);
+    // `dot_kernel` is consumed only by the `#[cfg(target_arch = "aarch64")]`
+    // fast paths below; discard it on other targets to avoid an unused-variable
+    // error under `-D warnings` (mirrors the sibling matmul helpers).
+    let _ = dot_kernel;
     let bits = 4usize;
     let layout = NBitsLayout { bits, block_size };
```
