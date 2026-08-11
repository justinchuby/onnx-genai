# Decision: CUDA EP B2 Fix — Device-ID Comparison for D2D Copies

**Author:** Batty (Systems)
**Date:** 2026-08-11
**Context:** PR #762, B2 was deferred by Nabil citing `MemoryDevice_GetDeviceId` might not exist. Gaff verified it does exist in ORT 1.27 bindings.

## Problem

Same-device D2D copies were rejected because the code compared `OrtMemoryDevice*` pointer equality. ORT may pass distinct pointers for the same physical device (device 0), causing functional failure for any model that moves tensors between subgraphs on the same GPU.

## Decision

Added `is_same_device()` helper in `transfer.rs` that:
1. Fast path: pointer equality → same device (no API call needed).
2. Null guard: either pointer null → fail closed (cross-device).
3. `MemoryDevice_GetDeviceId`: if `Some`, compare `u32` device IDs.
4. If `None` (function pointer not populated at runtime) → fail closed.

Applied to both `transfer_full_can_copy` and `transfer_full_copy_tensors`.

Cross-device (peer-to-peer) copies continue to fail closed with an actionable `OrtStatus` error.

## Verification

- 6 new unit tests covering: same-id/different-pointers, different-ids, missing API, pointer equality, null pointers, non-vacuity proof.
- `cargo test -p onnx-runtime-ep-plugin` — 161 passed (unit) + 9 passed (integration).
- `cargo test -p onnx-runtime-ep-cuda-plugin` — all pass.
- `cargo test -p onnx-runtime-ep-cpu-plugin` — all pass (no regression).
- `cargo clippy --all-targets -- -D warnings` on both EP crates — clean.
- `cargo fmt --check` — clean.

## What Remains Unknowable Without Hardware

- Whether ORT actually populates `MemoryDevice_GetDeviceId` at runtime (it's `Option<fn>`)
- Whether the device IDs returned match CUDA device ordinals
- Actual D2D copy performance on real hardware

Blocked on #768 for hardware validation.
